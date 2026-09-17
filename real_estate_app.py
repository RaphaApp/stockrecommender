from __future__ import annotations

"""
Japan Real Estate Quantitative Screener
=======================================
A single-file Streamlit application modeled on the Alpha Quant Engine stock
screener architecture:

  * tr() localization framework (English / 日本語) with English fallback
  * Streamlit 2025 compliance — strictly NO use_container_width=True; every
    widget that supports it uses width="stretch" instead
  * SQLite persistence (WAL mode) with `system_weights` + `historical_picks`
  * A quantitative scoring engine driven by dynamic, persisted KPI weights
  * An autonomous Walk-Forward feedback loop ("Run Market Reality Check")
    that records OBSERVED listing outcomes and re-tunes the scoring weights — fast
    sales push their dominant factors UP, stale/discounted listings push
    their dominant factors DOWN.

Run with:  streamlit run real_estate_app.py
"""

import base64
import contextlib
import gzip
import hashlib
import html
import json
import math
import os
import re
import sqlite3
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ----------------------------------------------------------------------------
# Global Configuration
# ----------------------------------------------------------------------------
# Honour an explicit DB_PATH from the environment. Without this the smoke test's
# claim of an isolated database was simply false — it set the variable and the
# app ignored it, so a test run wrote to the developer's real database. Secrets
# are consulted lazily elsewhere; this one must resolve at import time because
# get_conn() uses it immediately.
DB_PATH = os.environ.get("DB_PATH") or "real_estate_engine.db"

# The five scoring factors. Order matters for display only.
FACTORS = ["yield_score", "station_proximity", "asset_age", "price_efficiency",
           "development_potential"]

DEFAULT_WEIGHTS = {
    "yield_score": 0.30,           # gross yield is king for income property
    "station_proximity": 0.20,     # walk-to-station minutes drive liquidity in JP
    "asset_age": 0.15,             # building age (depreciation / earthquake code)
    "price_efficiency": 0.20,      # ¥/m² vs the prefecture benchmark (value gauge)
    "development_potential": 0.15, # 都市計画: zoning class + designated FAR (容積率)
}

MIN_WEIGHT = 0.05        # floor so no factor can ever be tuned to zero
LEARNING_RATE = 0.08     # step size of the walk-forward nudges
FAST_SALE_DAYS = 30      # sold under this -> the engine was "right"
STALE_DAYS = 120         # sat this long (or price cut) -> the engine was "wrong"
PREFECTURES = ["Tokyo", "Kanagawa", "Saitama", "Chiba", "Osaka", "Kyoto",
               "Hyogo", "Aichi", "Fukuoka"]

# Per-prefecture market parameters used by the mock generator so the synthetic
# listings feel realistic (Tokyo: pricier / lower yield; Chiba: cheaper / higher).
# sqm_rate is the benchmark ¥/m² the price-efficiency factor scores against.
PREF_PROFILES = {
    "Tokyo":    {"price_mu": 68_000_000, "sqm_rate": 1_050_000, "yield_mu": 5.2, "yield_sd": 1.1},
    "Kanagawa": {"price_mu": 45_000_000, "sqm_rate":   660_000, "yield_mu": 6.4, "yield_sd": 1.3},
    "Saitama":  {"price_mu": 33_000_000, "sqm_rate":   480_000, "yield_mu": 7.6, "yield_sd": 1.5},
    "Chiba":    {"price_mu": 28_000_000, "sqm_rate":   420_000, "yield_mu": 8.3, "yield_sd": 1.7},
    "Osaka":    {"price_mu": 38_000_000, "sqm_rate":   620_000, "yield_mu": 6.8, "yield_sd": 1.4},
    "Kyoto":    {"price_mu": 33_000_000, "sqm_rate":   540_000, "yield_mu": 6.5, "yield_sd": 1.3},
    # Built-in fallback estimates only — replaced by real MLIT medians on refresh.
    "Hyogo":    {"price_mu": 30_000_000, "sqm_rate":   480_000, "yield_mu": 6.9, "yield_sd": 1.4},
    "Aichi":    {"price_mu": 28_000_000, "sqm_rate":   450_000, "yield_mu": 7.1, "yield_sd": 1.4},
    "Fukuoka":  {"price_mu": 26_000_000, "sqm_rate":   420_000, "yield_mu": 7.6, "yield_sd": 1.5},
}

# Realistic floor-area ranges (m²) per property type; the mock draws uniformly
# inside the band and derives the asking price from area × prefecture ¥/m².
# ---------------------------------------------------------------------------
# Net-yield (NOI) model
# ---------------------------------------------------------------------------
# Gross yield (満室時利回り) is what portals advertise: annual rent at FULL
# occupancy over price, before any cost whatsoever. It systematically flatters
# high-fee buildings — two listings both advertising 5.6% can separate into
# 4.2% and 3.1% once the owner's real costs land. Net yield subtracts them.
# Every assumption here is a named constant, surfaced and tunable in the KPI
# Weights tab: the model is meant to be auditable, not a magic number.
FEE_PER_SQM_MONTH = 300.0     # 管理費+修繕積立金 estimate (¥/m²/month) used only
                              # when the listing's actual figure isn't supplied
OWNER_MAINT_PCT = 12.0        # 一棟/戸建: owner-borne maintenance + reserves as
                              # % of rent (no 管理組合 collecting monthly fees)
VACANCY_PCT = 5.0             # assumed vacancy (空室率)
MGMT_PCT = 5.0                # 賃貸管理手数料 as % of collected rent
TAX_PCT_OF_PRICE = 0.7        # 固定資産税+都市計画税, annual, as % of PRICE.
                              # (Statutory 1.4%+0.3% applies to the 固定資産税
                              # 評価額, which for a condo typically runs ~40–60%
                              # of market — hence ~0.7% of price, not 1.7%.)
# Types where the owner maintains the asset directly instead of paying monthly
# association fees.
OWNER_MAINTAINED_PTYPES = {"Whole Building Apartment", "Detached House"}

# ---------------------------------------------------------------------------
# 新耐震基準 (seismic standard) classification
# ---------------------------------------------------------------------------
# Japan's 新耐震基準 applies to building permits issued from 1981-06-01. This is
# not a cosmetic label: 旧耐震 stock is materially harder to finance (many banks
# decline it or shorten terms), has a thinner resale market, costs more to
# insure, and may face 耐震補強 expense. Treated as a scoring GATE, like the
# 市街化調整区域 zoning trap — a categorical risk, not a smooth factor.
SEISMIC_NEW_FROM = 1984       # completions from 1984 are reliably 新耐震
SEISMIC_GREY_FROM = 1982      # 1982–83 completions usually are, but the PERMIT
                              # may predate 1981-06-01 — worth verifying
OLD_SEISMIC_PENALTY = 10.0    # default points deducted for 旧耐震 (tunable)

# ---------------------------------------------------------------------------
# Yield basis — which figure the listing is actually quoting
# ---------------------------------------------------------------------------
# 満室時利回り / 想定利回り / 表面利回り are all AT FULL OCCUPANCY: they assume
# every unit is let, so a vacancy allowance must be subtracted to get a real
# net figure. 現況利回り is the CURRENT actual figure — it already reflects the
# building's real occupancy, so deducting vacancy again double-counts it and
# understates the yield. The basis therefore has to travel with the number.
YIELD_BASES = ["full", "current", "unknown"]
YIELD_BASIS_DEFAULT = "full"        # what portals quote most often, and the
                                    # conservative choice (vacancy IS deducted)

# ---------------------------------------------------------------------------
# Acquisition costs (諸費用) — the gap between price and invested capital
# ---------------------------------------------------------------------------
# Every yield a portal quotes is yield ON PRICE. What an owner actually invests
# is price + closing costs, which in Japan runs ~5–8%: brokerage (3% + ¥60,000
# + consumption tax), 不動産取得税, 登録免許税, 司法書士報酬, 印紙税, fire
# insurance and the 固定資産税 settlement. Brokerage is computed formulaically
# (its fixed ¥60,000 component bites harder on cheap units); the rest is a
# tunable percentage.
BROKERAGE_PCT = 3.0           # 仲介手数料 rate on price
BROKERAGE_FIXED = 60_000.0    # the "+6万円" that makes cheap units relatively worse
CONSUMPTION_TAX_PCT = 10.0    # 消費税 on the brokerage fee
ACQ_OTHER_PCT = 2.5           # 取得税+登録免許税+司法書士+印紙+保険 etc., % of price

# ---------------------------------------------------------------------------
# Floor-area liquidity thresholds (exit risk / obsolescence)
# ---------------------------------------------------------------------------
# Floor area is not just a price driver — in Japan it gates who can BUY the unit
# from you later, which is exit liquidity:
#   <20m²  投資用ローン is commonly unavailable (many lenders set a 20m² or 25m²
#          minimum), leaving a cash-buyer-only exit and a wide bid-ask spread.
#   <25m²  Some lenders decline, and many Tokyo wards' ワンルームマンション
#          ordinances set ~25m² as the minimum for NEW builds — which cuts both
#          ways: existing small stock gets scarcer, but is regulatorily disfavoured.
#   <50m²  Below the 住宅ローン控除 / acquisition-tax reduction threshold, so
#          owner-occupiers are effectively excluded: investors only. Normal for
#          this asset class, so it's a NOTE, not a penalty.
# Counterweight worth stating: single-person households are Japan's fastest-
# growing household type, which structurally supports small-unit RENTAL demand.
# The risk small units carry is about financing and resale, not tenant demand.
AREA_LOAN_FLOOR = 20.0
AREA_LENDER_CAUTION = 25.0
AREA_OWNER_OCC = 50.0
SMALL_UNIT_PENALTY = 8.0      # default points for sub-20m² (tunable)

# ---------------------------------------------------------------------------
# Resale outlook — official population projections (XKT013, 500m mesh)
# ---------------------------------------------------------------------------
# The exit-liquidity thresholds above are structural (loan floors, tax rules) but
# STATIC. What actually decides whether a unit is sellable in 10–15 years is
# whether anyone is still living there to buy it. MLIT serves the government's
# 将来推計人口 on a 500m mesh — local enough to describe a genuine catchment,
# and FORWARD-looking, unlike our backward-looking price momentum.
#
# So the small-unit penalty is SCALED by the demographic outlook: a compact unit
# in a growing catchment is a very different asset from the same unit in a
# catchment projected to lose a fifth of its people.
#
# Honest limits: projections are made at a fixed vintage and cannot know about
# future redevelopment, so a mesh containing a major project can outperform its
# own projection. They describe population, not household COUNT — and because
# Japanese household sizes keep shrinking, household count (the real driver of
# housing demand) falls more slowly than population does. Read it as a direction,
# not a forecast.
POP_ENDPOINT = "https://www.reinfolib.mlit.go.jp/ex-api/external/XKT013"
RESALE_HORIZON_YEARS = 15     # how far out to read the projection (tunable)
RESALE_DECLINE_SEVERE = -15.0 # % change at/below which resale risk compounds hard
RESALE_DECLINE_MILD = -5.0
RESALE_GROWTH = 5.0           # % change at/above which the catchment supports resale
RESALE_MULTIPLIERS = {"severe": 1.5, "mild": 1.25, "flat": 1.0, "growth": 0.75}
# Sanity guards. A projection is only trustworthy if the catchment is populated
# enough for a percentage to mean anything, and if the result is physically
# plausible. Japan's worst-projected municipalities lose roughly a third over 15
# years; a tile-level figure beyond RESALE_MAX_PLAUSIBLE is a data artifact, not
# a forecast, and must be reported as such rather than scored.
POP_MIN_BASE = 300            # people in the ~1km tile; below this a % is noise
POP_MIN_ABS_LOSS = 150        # absolute loss needed before "severe" is credible
RESALE_MAX_PLAUSIBLE = 50.0   # |%| beyond this is treated as an anomaly

PORTAL_DOMAINS = {
    "LIFULL HOME'S": "www.homes.co.jp",
    "Yahoo Real Estate": "realestate.yahoo.co.jp",
    "Kenbiya": "www.kenbiya.com",
    "Rakumachi": "www.rakumachi.jp",
    "Renosy": "www.renosy.com",
}

# Portals offered in the MANUAL analyzer (screenshot/typed entry). Renosy blocks
# scrapers, which is exactly why it appears here and not in the scan portals.
MANUAL_PORTALS = ["Kenbiya", "Rakumachi", "Renosy"]

# Maps from vision-extracted Japanese text to internal keys.
# ORDER MATTERS: "京都" is a substring of "東京都". Matching iterates in dict
# order, so 東京 MUST precede 京都 or every Tokyo address would resolve to Kyoto.
PREF_JP_MAP = {"東京": "Tokyo", "神奈川": "Kanagawa", "埼玉": "Saitama",
               "千葉": "Chiba", "大阪": "Osaka", "京都": "Kyoto",
               "兵庫": "Hyogo", "愛知": "Aichi", "福岡": "Fukuoka"}

# URL slug -> internal keys. Portal URLs are surprisingly information-dense:
#   kenbiya.com/pp1/s/<pref-romaji>/<city>/re_<id>/
#   rakumachi.jp/syuuekibukken/<region>/<pref-romaji>/dim<code>/<id>/show.html
PREF_SLUG_MAP = {"tokyo": "Tokyo", "kanagawa": "Kanagawa",
                 "saitama": "Saitama", "chiba": "Chiba", "osaka": "Osaka",
                 "kyoto": "Kyoto", "hyogo": "Hyogo", "kobe": "Hyogo",
                 "aichi": "Aichi", "nagoya": "Aichi", "fukuoka": "Fukuoka"}
# Rakumachi dim-codes observed in the wild; only codes we've actually verified
# are mapped — unknown codes set nothing rather than guessing.
RAKUMACHI_DIM_PTYPE = {
    "dim1002": "Whole Building Apartment",   # 一棟アパート
    # "dim2001" = 区分マンション (sectional condo unit): the unit layout (1R/1K/
    # 2LDK…) is NOT derivable from the code, so we deliberately set no ptype.
}
LAYOUT_TO_PTYPE = [   # (substring match on the extracted layout, in priority order)
    # CRITICAL ORDERING: this is a first-match-wins substring scan, and "LDK"
    # is itself a substring of "1LDK"/"2LDK"/"3LDK" — so every specific NLDK
    # entry MUST be listed before the bare "LDK" fallback, or the generic rule
    # fires first and every layout (a studio 1LDK or a family 3LDK alike)
    # collapses into one bucket. (Verified: none of 1K/1DK/1LDK/2K/2DK/2LDK/
    # 3DK/3LDK are substrings of each other, so their relative order is safe —
    # only the bare "LDK"/"K" fallbacks need to stay last.)
    ("1R", "1R Mansion"),
    ("1DK", "1DK Apartment"), ("1LDK", "1LDK Apartment"), ("1K", "1K Apartment"),
    ("2DK", "2DK Mansion"), ("2LDK", "2LDK Mansion"), ("2K", "2K Apartment"),
    ("3DK", "3DK Mansion"), ("3LDK", "3LDK Mansion"),
    ("LDK", "3LDK Mansion"),   # unrecognized/4LDK+ variant -> family-size bucket
    ("一棟", "Whole Building Apartment"),
    ("WHOLE", "Whole Building Apartment"), ("戸建", "Detached House"),
]

# JIS prefecture codes used by the MLIT 不動産情報ライブラリ API.
PREF_CODES = {"Saitama": "11", "Chiba": "12", "Tokyo": "13", "Kanagawa": "14",
              "Osaka": "27", "Kyoto": "26", "Hyogo": "28", "Aichi": "23", "Fukuoka": "40"}
# Inverse: the 2-digit JIS prefecture code -> internal prefecture key. A 5-digit
# 市区町村コード is just <pref-code><3-digit-municipality>, so its first two
# characters map a municipality back to its prefecture for the fallback tiers.
PREF_CODE_TO_KEY = {code: key for key, code in PREF_CODES.items()}

# ----------------------------------------------------------------------------
# Municipality (市区町村) code resolution layer — the hyper-local benchmark key
# ----------------------------------------------------------------------------
# The whole upgrade hinges on resolving a listing to its 5-digit MLIT 市区町村
# コード (JIS X 0402) so a Shibuya unit is benchmarked against Shibuya sales —
# not a Tokyo-wide blend that luxury central-ward prices distort. This is a
# static lookup (no network) keyed per prefecture by the municipality's
# canonical Japanese name FIRST, with romaji aliases following so URL slugs and
# vision-extracted address strings both resolve. CITY_CODE_TO_NAME (built below)
# takes the FIRST (Japanese) entry per code as the display name, so ordering
# matters: always list the Japanese name before any romaji alias.
MUNICIPALITY_CODES: dict[str, dict[str, str]] = {
    # --- Tokyo 23 special wards (13101–13123, in official order) -------------
    "Tokyo": {
        "千代田区": "13101", "chiyoda": "13101",
        "中央区": "13102", "chuo": "13102",
        "港区": "13103", "minato": "13103",
        "新宿区": "13104", "shinjuku": "13104",
        "文京区": "13105", "bunkyo": "13105",
        "台東区": "13106", "taito": "13106",
        "墨田区": "13107", "sumida": "13107",
        "江東区": "13108", "koto": "13108",
        "品川区": "13109", "shinagawa": "13109",
        "目黒区": "13110", "meguro": "13110",
        "大田区": "13111", "ota": "13111",
        "世田谷区": "13112", "setagaya": "13112",
        "渋谷区": "13113", "shibuya": "13113",
        "中野区": "13114", "nakano": "13114",
        "杉並区": "13115", "suginami": "13115",
        "豊島区": "13116", "toshima": "13116",
        "北区": "13117", "kita": "13117",
        "荒川区": "13118", "arakawa": "13118",
        "板橋区": "13119", "itabashi": "13119",
        "練馬区": "13120", "nerima": "13120",
        "足立区": "13121", "adachi": "13121",
        "葛飾区": "13122", "katsushika": "13122",
        "江戸川区": "13123", "edogawa": "13123",
        # A couple of high-volume Tama-area cities investors actually screen.
        "武蔵野市": "13203", "musashino": "13203",
        # NOTE 府中市 also exists in Hiroshima (34210); the resolver is
        # prefecture-scoped, so this Tokyo entry is unambiguous here.
        "府中市": "13206", "fuchu": "13206",
        "調布市": "13208", "chofu": "13208",
        "町田市": "13209", "machida": "13209",
        "三鷹市": "13204", "mitaka": "13204",
    },
    # --- Kanagawa: Yokohama / Kawasaki / Sagamihara wards + key cities -------
    "Kanagawa": {
        "横浜市鶴見区": "14101",
        "横浜市神奈川区": "14102",
        "横浜市西区": "14103", "yokohama-nishi": "14103",
        "横浜市中区": "14104", "yokohama-naka": "14104",
        "横浜市南区": "14105",
        "横浜市保土ケ谷区": "14106",
        "横浜市磯子区": "14107",
        "横浜市金沢区": "14108",
        "横浜市港北区": "14109", "kohoku": "14109",
        "横浜市戸塚区": "14110",
        "横浜市港南区": "14111",
        "横浜市旭区": "14112",
        "横浜市緑区": "14113",
        "横浜市瀬谷区": "14114",
        "横浜市栄区": "14115",
        "横浜市泉区": "14116",
        "横浜市青葉区": "14117", "aoba": "14117",
        "横浜市都筑区": "14118", "tsuzuki": "14118",
        "川崎市川崎区": "14131", "kawasaki": "14131",
        "川崎市幸区": "14132",
        "川崎市中原区": "14133", "nakahara": "14133",
        "川崎市高津区": "14134",
        "川崎市多摩区": "14135",
        "川崎市宮前区": "14136",
        "川崎市麻生区": "14137",
        "相模原市緑区": "14151",
        "相模原市中央区": "14152",
        "相模原市南区": "14153",
        "横須賀市": "14201", "yokosuka": "14201",
        # 大船 (Ofuna) is a DISTRICT, not a municipality: it straddles 鎌倉市 and
        # 横浜市栄区 (14115, already listed). An Ofuna address resolves to
        # whichever of those two it actually sits in.
        "鎌倉市": "14204", "kamakura": "14204",
        "藤沢市": "14205", "fujisawa": "14205",
        "厚木市": "14210", "atsugi": "14210",
        "大和市": "14213", "yamato": "14213",
    },
    # --- Chiba: Chiba-shi wards + the Tokyo-bay commuter-belt hubs -----------
    "Chiba": {
        "千葉市中央区": "12101", "chiba-chuo": "12101",
        "千葉市花見川区": "12102",
        "千葉市稲毛区": "12103",
        "千葉市若葉区": "12104",
        "千葉市緑区": "12105",
        "千葉市美浜区": "12106", "mihama": "12106",
        "市川市": "12203", "ichikawa": "12203",
        "船橋市": "12204", "funabashi": "12204",
        "松戸市": "12207", "matsudo": "12207",
        "習志野市": "12216", "narashino": "12216",
        "柏市": "12217", "kashiwa": "12217",
        "浦安市": "12227", "urayasu": "12227",
    },
    # --- Saitama: Saitama-shi wards + the Saikyo/Tobu-line commuter cities ---
    "Saitama": {
        "さいたま市西区": "11101",
        "さいたま市北区": "11102",
        "さいたま市大宮区": "11103", "omiya": "11103",
        "さいたま市見沼区": "11104",
        "さいたま市中央区": "11105",
        "さいたま市桜区": "11106",
        "さいたま市浦和区": "11107", "urawa": "11107",
        "さいたま市南区": "11108",
        "さいたま市緑区": "11109",
        "さいたま市岩槻区": "11110",
        "川越市": "11201", "kawagoe": "11201",
        "熊谷市": "11202", "kumagaya": "11202",
        "川口市": "11203", "kawaguchi": "11203",
        "所沢市": "11208", "tokorozawa": "11208",
    },
    # --- Osaka: 大阪市 24 wards (27101–27128) + Sakai wards & key hub cities ---
    "Osaka": {
        "大阪市都島区": "27102", "miyakojima": "27102",
        "大阪市福島区": "27103", "fukushima-osaka": "27103",
        "大阪市此花区": "27104",
        "大阪市西区": "27106", "osaka-nishi": "27106",
        "大阪市港区": "27107", "minato-osaka": "27107",
        "大阪市大正区": "27108",
        "大阪市天王寺区": "27109", "tennoji": "27109",
        "大阪市浪速区": "27111", "naniwa": "27111",
        "大阪市西淀川区": "27113",
        "大阪市東淀川区": "27114",
        "大阪市東成区": "27115",
        "大阪市生野区": "27116",
        "大阪市旭区": "27117",
        "大阪市城東区": "27118",
        "大阪市阿倍野区": "27119", "abeno": "27119",
        "大阪市住吉区": "27120",
        "大阪市東住吉区": "27121",
        "大阪市西成区": "27122",
        "大阪市淀川区": "27123", "yodogawa": "27123",
        "大阪市鶴見区": "27124",
        "大阪市住之江区": "27125",
        "大阪市平野区": "27126",
        "大阪市北区": "27127", "kita-osaka": "27127",
        "大阪市中央区": "27128", "chuo-osaka": "27128",
        # Key non-Osaka-shi hubs an investor screens.
        "堺市堺区": "27141", "sakai": "27141",
        "豊中市": "27203", "toyonaka": "27203",
        "吹田市": "27205", "suita": "27205",
        "高槻市": "27207", "takatsuki": "27207",
        "東大阪市": "27227", "higashiosaka": "27227",
    },
    # --- Kyoto: 京都市 11 wards (26101–26111) + Uji ---------------------------
    "Kyoto": {
        "京都市北区": "26101", "kyoto-kita": "26101",
        "京都市上京区": "26102", "kamigyo": "26102",
        "京都市左京区": "26103", "sakyo": "26103",
        "京都市中京区": "26104", "nakagyo": "26104",
        "京都市東山区": "26105", "higashiyama": "26105",
        "京都市下京区": "26106", "shimogyo": "26106",
        "京都市南区": "26107", "kyoto-minami": "26107",
        "京都市右京区": "26108", "ukyo": "26108",
        "京都市伏見区": "26109", "fushimi": "26109",
        "京都市山科区": "26110", "yamashina": "26110",
        "京都市西京区": "26111", "nishikyo": "26111",
        "宇治市": "26204", "uji": "26204",
    },
    # --- Hyogo: 神戸市 9 wards (28103/28104 no longer exist — merged into 中央区
    #     in 1980) + the Hanshin corridor cities investors actually screen ------
    "Hyogo": {
        "神戸市東灘区": "28101", "higashinada": "28101",
        "神戸市灘区": "28102", "nada": "28102",
        "神戸市兵庫区": "28105", "kobe-hyogo": "28105",
        "神戸市長田区": "28106", "nagata": "28106",
        "神戸市須磨区": "28107", "suma": "28107",
        "神戸市垂水区": "28108", "tarumi": "28108",
        "神戸市北区": "28109", "kobe-kita": "28109",
        "神戸市中央区": "28110", "kobe-chuo": "28110",
        "神戸市西区": "28111", "kobe-nishi": "28111",
        "尼崎市": "28202", "amagasaki": "28202",
        "西宮市": "28204", "nishinomiya": "28204",
        "芦屋市": "28206", "ashiya": "28206",
    },
    # --- Aichi: 名古屋市 16 wards ------------------------------------------------
    "Aichi": {
        "名古屋市千種区": "23101", "chikusa": "23101",
        "名古屋市東区": "23102", "nagoya-higashi": "23102",
        "名古屋市北区": "23103", "nagoya-kita": "23103",
        "名古屋市西区": "23104", "nagoya-nishi": "23104",
        "名古屋市中村区": "23105", "nakamura": "23105",
        "名古屋市中区": "23106", "nagoya-naka": "23106",
        "名古屋市昭和区": "23107", "showa": "23107",
        "名古屋市瑞穂区": "23108", "mizuho": "23108",
        "名古屋市熱田区": "23109", "atsuta": "23109",
        "名古屋市中川区": "23110", "nakagawa": "23110",
        "名古屋市港区": "23111", "nagoya-minato": "23111",
        "名古屋市南区": "23112", "nagoya-minami": "23112",
        "名古屋市守山区": "23113", "moriyama": "23113",
        "名古屋市緑区": "23114", "nagoya-midori": "23114",
        "名古屋市名東区": "23115", "meito": "23115",
        "名古屋市天白区": "23116", "tempaku": "23116",
    },
    # --- Fukuoka: 北九州市 7 wards (designated 1963, hence the lower codes) +
    #     福岡市 7 wards (designated 1972) -------------------------------------
    "Fukuoka": {
        "北九州市門司区": "40101", "moji": "40101",
        "北九州市若松区": "40103", "wakamatsu": "40103",
        "北九州市戸畑区": "40105", "tobata": "40105",
        "北九州市小倉北区": "40106", "kokura-kita": "40106",
        "北九州市小倉南区": "40107", "kokura-minami": "40107",
        "北九州市八幡東区": "40108", "yahata-higashi": "40108",
        "北九州市八幡西区": "40109", "yahata-nishi": "40109",
        "福岡市東区": "40131", "fukuoka-higashi": "40131",
        "福岡市博多区": "40132", "hakata": "40132",
        "福岡市中央区": "40133", "fukuoka-chuo": "40133",
        "福岡市南区": "40134", "fukuoka-minami": "40134",
        "福岡市西区": "40135", "fukuoka-nishi": "40135",
        "福岡市城南区": "40136", "jonan": "40136",
        "福岡市早良区": "40137", "sawara": "40137",
    },
}

# Bounded set of "target investment cities" the MLIT refresh actually queries
# per prefecture (one API call per municipality per quarter — see
# fetch_mlit_benchmarks). Kept deliberately tighter than MUNICIPALITY_CODES
# above (which is a broad RESOLUTION table): refreshing every ward × 4 quarters
# would be a needless API storm, so we hit the central wards / commuter hubs an
# income investor screens. Resolution still works for any municipality in the
# table above; non-target ones simply fall back to the prefecture benchmark.
MLIT_TARGET_CITIES: dict[str, list[str]] = {
    "Tokyo":    ["13101", "13102", "13103", "13104", "13105", "13107", "13108",
                 "13109", "13110", "13111", "13112", "13113", "13114", "13115",
                 "13116", "13117", "13119", "13120", "13121",
                 "13203", "13206", "13208", "13209"],   # 武蔵野/府中/調布/町田
                 # central business/premium + common residential investor wards
                 # (Setagaya, Suginami, Nakano, Ota, Kita, Itabashi, Nerima, Adachi…)
    "Kanagawa": ["14103", "14104", "14109", "14117", "14131", "14133", "14134",
                 "14204", "14205"],   # incl. Kawasaki Nakahara/Takatsu + 鎌倉(大船)
    "Chiba":    ["12101", "12106", "12204", "12216", "12227"],
    "Saitama":  ["11103", "11107", "11203", "11208"],
    "Osaka":    ["27102", "27103", "27104", "27106", "27107", "27108", "27109",
                 "27111", "27113", "27114", "27115", "27116", "27117", "27118",
                 "27119", "27120", "27121", "27122", "27123", "27124", "27125",
                 "27126", "27127", "27128",              # all 24 大阪市 wards
                 "27141", "27203", "27205", "27227"],    # Sakai-ku, Toyonaka, Suita, Higashi-Osaka
    "Kyoto":    ["26101", "26102", "26103", "26104", "26105", "26106",
                 "26107", "26108", "26109", "26110", "26111",   # all 11 京都市 wards
                 "26204"],                                       # 宇治市
    "Hyogo":    ["28110", "28101", "28102", "28105", "28204", "28202"],
                 # 神戸中央/東灘/灘/兵庫 + 西宮・尼崎 (Hanshin corridor)
    "Aichi":    ["23106", "23105", "23102", "23101", "23107", "23115"],
                 # 名古屋 中/中村(駅前)/東/千種/昭和/名東
    "Fukuoka":  ["40132", "40133", "40131", "40134", "40137", "40106"],
                 # 博多/中央(天神)/東/南/早良 + 小倉北 (Kitakyushu)
}

# Inverse map (5-digit code -> canonical Japanese name) for display in the
# sidebar / analyzer. Built from the FIRST Japanese name listed per code.
CITY_CODE_TO_NAME: dict[str, str] = {}
for _pref_key, _table in MUNICIPALITY_CODES.items():
    for _name, _code in _table.items():
        if not _name.isascii() and _code not in CITY_CODE_TO_NAME:
            CITY_CODE_TO_NAME[_code] = _name

# Accept a prefecture as an internal key ("Tokyo"), a bare Japanese name ("東京")
# or a full Japanese name with the 都/県 suffix ("東京都"), so
# resolve_municipality_code copes with whatever the upstream extractor produced.
# (PREF_NAMES is defined further below, so the 都/県 forms are derived here
# directly from PREF_JP_MAP rather than referencing it.)
_PREF_NAME_TO_KEY: dict[str, str] = {}
for _k in PREFECTURES:
    _PREF_NAME_TO_KEY[_k] = _k                                  # "Tokyo" -> "Tokyo"
for _jp, _k in PREF_JP_MAP.items():
    _PREF_NAME_TO_KEY[_jp] = _k                                 # "東京"   -> "Tokyo"
    # Suffix is prefecture-specific: 東京都 / 大阪府 / everything-else県.
    _suffix = {"東京": "都", "大阪": "府", "京都": "府"}.get(_jp, "県")
    _full = _jp + _suffix                                       # "東京都"/"大阪府"/"神奈川県"…
    _PREF_NAME_TO_KEY[_full] = _k

def _strip_muni_suffix(name: str) -> str:
    """Drop a single trailing 区/市/町/村 so 'X区' and 'X' compare equal — the
    suffix-insensitive fallback used when no exact municipality match is found."""
    n = name.strip()
    return n[:-1] if n[-1:] in ("区", "市", "町", "村") else n

def _pref_key_from_city_code(city_code: str | None) -> str | None:
    """Recover the internal prefecture key from a 5-digit 市区町村コード via its
    leading 2-digit JIS prefecture code (e.g. '13113' -> '13' -> 'Tokyo')."""
    if not city_code:
        return None
    return PREF_CODE_TO_KEY.get(str(city_code)[:2])

def municipality_label(city_code: str | None) -> str:
    """Human-readable label for a 市区町村コード — the canonical Japanese name
    when known, else the bare code. Used in the benchmark console and analyzer."""
    if not city_code:
        return "—"
    return CITY_CODE_TO_NAME.get(str(city_code), str(city_code))

def _build_bare_ward_index() -> dict[str, dict[str, str]]:
    """Bare designated-city ward names that are UNAMBIGUOUS inside their own
    prefecture, e.g. {"Fukuoka": {"博多区": "40132", ...}}.

    The table stores designated-city wards as compounds (福岡市博多区) because
    the same ward name recurs nationwide — there are four 中央区 and five 北区
    across the covered prefectures. But lookups are ALWAYS prefecture-scoped, and
    within one prefecture most bare names are unique, so refusing them threw away
    real resolution: a Fukuoka listing saying just 「博多区」 fell all the way back
    to the prefecture average.

    Derived at import rather than hand-maintained, so it stays correct as cities
    are added. Names that collide inside a prefecture (Kanagawa's 南区 and 緑区
    exist in BOTH Yokohama and Sagamihara) are deliberately EXCLUDED — those must
    stay unresolved rather than silently pick a side."""
    index: dict[str, dict[str, str]] = {}
    for pref, table in MUNICIPALITY_CODES.items():
        seen: dict[str, set] = {}
        for full, code in table.items():
            if full.isascii() or "市" not in full or full.endswith("市"):
                continue
            ward = full.split("市", 1)[1]
            if ward:
                seen.setdefault(ward, set()).add(code)
        index[pref] = {w: next(iter(c)) for w, c in seen.items() if len(c) == 1}
    return index

_BARE_WARD_INDEX = _build_bare_ward_index()

def resolve_municipality_code(pref_name: str, city_name: str) -> str | None:
    """Resolve a (prefecture, municipality) pair to the standard 5-digit MLIT
    市区町村コード, or None when it can't be confidently placed.

    `pref_name` may be an internal key ('Tokyo'), a bare ('東京') or full
    ('東京都') Japanese name. `city_name` may be a clean municipality ('渋谷区',
    '習志野市', '横浜市西区'), a romaji slug ('shibuya', 'yokohama-nishi'), or a
    longer address fragment that merely CONTAINS the municipality
    ('東京都渋谷区神宮前1-2-3'). Matching runs strict -> loose so the most
    specific hit wins:
        1. exact name match
        2. (Japanese) substring match, longest canonical name first — catches
           full-address fragments and designated-city ward compounds
        3. suffix-insensitive match (strip 区/市/町/村) — the robust fallback
    Returning None (rather than guessing) lets callers degrade cleanly to the
    prefecture benchmark instead of mis-pricing against the wrong ward."""
    if not city_name:
        return None
    pref_key = _PREF_NAME_TO_KEY.get(str(pref_name), str(pref_name))
    table = MUNICIPALITY_CODES.get(pref_key)
    if not table:
        return None
    q = unicodedata.normalize("NFKC", str(city_name)).strip().replace(" ", "").replace("\u3000", "")
    if not q:
        return None
    # Romaji path: EXACT match only — substring matching on ASCII risks false
    # hits ('kita' inside 'kitamoto'); tolerate a trailing -ku / -shi / -machi.
    if q.isascii():
        ql = q.lower().strip("-")
        # Strip a HYPHEN-DELIMITED administrative suffix only. The earlier code
        # also carried bare "ku"/"shi", which matched on spelling rather than a
        # suffix token and corrupted any ward ENDING in those letters —
        # "yokohama-nishi" -> stripped "shi" -> "yokohama-ni" -> missed its own
        # alias. Table aliases are already bare stems ("shibuya", not
        # "shibuyaku"), so hyphen-delimited stripping is sufficient and safe.
        for suf in ("-ku", "-shi", "-machi", "-cho", "-mura"):
            if ql.endswith(suf) and len(ql) > len(suf):
                ql = ql[: -len(suf)].strip("-")
                break
        for name, code in table.items():
            if name.isascii() and name.lower() == ql:
                return code
        # Last-chance: a hyphenless compound like "shibuyaku" — accept ONLY if
        # the de-suffixed remainder is itself a real alias (never a blind strip).
        for suf in ("ku", "shi", "machi", "cho", "mura"):
            if ql.endswith(suf) and len(ql) > len(suf):
                stem = ql[: -len(suf)]
                for name, code in table.items():
                    if name.isascii() and name.lower() == stem:
                        return code
                break
        return None
    # Japanese path.
    if q in table:                                   # 1. exact
        return table[q]
    jp_names = sorted((n for n in table if not n.isascii()), key=len, reverse=True)
    for name in jp_names:                            # 2. substring, most specific first
        if name in q:
            return table[name]
    # 3. Bare designated-city ward name, unique within THIS prefecture.
    #    Placed after the compound tiers so a full 「横浜市西区」 always wins, and
    #    restricted to unambiguous names so 「南区」 in Kanagawa (Yokohama AND
    #    Sagamihara) still resolves to nothing rather than to a coin flip.
    bare_index = _BARE_WARD_INDEX.get(pref_key, {})
    if q in bare_index:
        return bare_index[q]
    for ward, code in sorted(bare_index.items(), key=lambda kv: -len(kv[0])):
        if ward in q:                                # address fragment containing it
            return code
    bare_q = _strip_muni_suffix(q)                   # 4. suffix-insensitive fallback
    for name in jp_names:
        if _strip_muni_suffix(name) == bare_q:
            return table[name]
    return None

# MLIT Real Estate Information Library — real-transaction-price endpoint.
# Free API key from https://www.reinfolib.mlit.go.jp (registration required).
# NOTE: verify the endpoint/params against the current official docs before
# production use; the API launched in 2024 and may evolve.
MLIT_ENDPOINT = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"
# Authoritative 市区町村 list per prefecture. Used to VERIFY our hard-coded JIS
# codes: a wrong code is otherwise a silent failure (that ward just returns no
# samples forever, or — worse — a different ward's data).
MLIT_CITYLIST_ENDPOINT = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT002"
MLIT_MIN_SAMPLES = 20            # below this, keep the built-in estimate
MLIT_MIN_QTR_SAMPLES = 8         # a QUARTERLY median needs this many samples to
                                 # count toward momentum — thin wards otherwise
                                 # show median noise as a "trend"
MLIT_PPSM_BOUNDS = (80_000, 3_000_000)   # sanity band for ¥/m² outlier rejection
MLIT_REQUEST_PAUSE_S = 0.25      # polite pause between the 16 quarter/pref calls

# Manual-analyzer verdict thresholds on the 100-point composite.
# Verdict bands. These are judgement, not physics, so they are DEFAULTS and are
# tunable in the Weights tab. Lowered from 70/55: in practice real listings score
# in the mid-40s to low-60s, so a 55 bar labelled almost everything a decline and
# the label stopped carrying information.
VERDICT_STRONG = 65.0
VERDICT_CONSIDER = 50.0

def get_verdict_strong() -> float:
    return float(st.session_state.get("verdict_strong", VERDICT_STRONG))

def get_verdict_consider() -> float:
    return float(st.session_state.get("verdict_consider", VERDICT_CONSIDER))

# ----------------------------------------------------------------------------
# 都市計画 (urban planning) inputs — zoning class + designated FAR (容積率)
# ----------------------------------------------------------------------------
# Both Kenbiya and Rakumachi print 用途地域 and 建ぺい率・容積率 in the 物件概要
# table, so these are user-readable today (manual analyzer) and adapter-parseable
# later. The MLIT 不動産情報ライブラリ also serves 都市計画決定GISデータ, but as
# coordinate-based tile queries — automated lookup belongs to a later step once
# listings carry lat/lon (geocoding).
#
# Base scores reflect REBUILD/INCOME flexibility for an investor, not "niceness":
# commercial zones allow the most floor area and uses; exclusive low-rise zones
# are restrictive; 工業専用地域 forbids housing outright; and 市街化調整区域
# (urbanization control areas) severely restrict rebuilding and financing — the
# classic trap behind suspiciously cheap high-yield exurban listings.
ZONING_KEYS = [
    "shogyo",            # 商業地域 — commercial
    "kinrin_shogyo",     # 近隣商業地域 — neighborhood commercial
    "jun_kogyo",         # 準工業地域 — quasi-industrial (very flexible for housing)
    "dai1_jukyo",        # 第一種住居地域
    "dai2_jukyo",        # 第二種住居地域
    "jun_jukyo",         # 準住居地域
    "dai1_chukoso",      # 第一種中高層住居専用地域
    "dai2_chukoso",      # 第二種中高層住居専用地域
    "dai1_teiso",        # 第一種低層住居専用地域
    "dai2_teiso",        # 第二種低層住居専用地域
    "den_jukyo",         # 田園住居地域
    "kogyo",             # 工業地域 — industrial (housing allowed, environment poor)
    "kogyo_senyo",       # 工業専用地域 — industrial-exclusive (housing FORBIDDEN)
    "choseikuiki",       # 市街化調整区域 — urbanization control area (rebuild restricted)
    "unknown",
]

ZONING_BASE_SCORES = {
    "shogyo": 92.0, "kinrin_shogyo": 82.0, "jun_kogyo": 75.0,
    "dai1_jukyo": 65.0, "dai2_jukyo": 65.0, "jun_jukyo": 60.0,
    "dai1_chukoso": 55.0, "dai2_chukoso": 55.0,
    "dai1_teiso": 40.0, "dai2_teiso": 45.0, "den_jukyo": 35.0,
    "kogyo": 45.0, "kogyo_senyo": 5.0, "choseikuiki": 8.0,
    "unknown": 50.0,   # no information -> neutral midpoint, never a penalty
}

# Typical designated-FAR (容積率 %) bands per zoning class, used by the mock
# generator so synthetic listings are internally consistent.
ZONING_FAR_BANDS = {
    "shogyo": (400, 700), "kinrin_shogyo": (200, 400), "jun_kogyo": (200, 400),
    "dai1_jukyo": (200, 300), "dai2_jukyo": (200, 300), "jun_jukyo": (200, 300),
    "dai1_chukoso": (150, 300), "dai2_chukoso": (150, 300),
    "dai1_teiso": (80, 150), "dai2_teiso": (100, 200), "den_jukyo": (100, 200),
    "kogyo": (200, 400), "kogyo_senyo": (200, 400), "choseikuiki": (100, 200),
}

# Mock zoning mix per prefecture (denser core -> more commercial; exurbs carry a
# real tail of 調整区域 listings, which is exactly the trap this factor catches).
PROPERTY_TYPES = [
    "1R Mansion", "1K Apartment", "1DK Apartment", "1LDK Apartment",
    "2K Apartment", "2DK Mansion", "2LDK Mansion",
    "3DK Mansion", "3LDK Mansion",
    "Whole Building Apartment", "Detached House", "Office Unit", "Corner Retail Unit",
]

# ----------------------------------------------------------------------------
# Localization Framework — tr()
# ----------------------------------------------------------------------------
LANGUAGES = {"English": "en", "日本語": "ja"}

PREF_NAMES = {
    "en": {"Tokyo": "Tokyo", "Kanagawa": "Kanagawa", "Saitama": "Saitama", "Chiba": "Chiba",
           "Osaka": "Osaka", "Kyoto": "Kyoto", "Hyogo": "Hyogo (Kobe)", "Aichi": "Aichi (Nagoya)",
           "Fukuoka": "Fukuoka"},
    "ja": {"Tokyo": "東京都", "Kanagawa": "神奈川県", "Saitama": "埼玉県", "Chiba": "千葉県",
           "Osaka": "大阪府", "Kyoto": "京都府", "Hyogo": "兵庫県", "Aichi": "愛知県",
           "Fukuoka": "福岡県"},
}

FACTOR_NAMES = {
    "en": {"yield_score": "Yield Score", "station_proximity": "Station Proximity",
           "asset_age": "Asset Age", "price_efficiency": "Price per m²",
           "development_potential": "Development Potential"},
    "ja": {"yield_score": "利回りスコア", "station_proximity": "駅近スコア",
           "asset_age": "築年数スコア", "price_efficiency": "㎡単価スコア",
           "development_potential": "開発ポテンシャル"},
}

ZONING_NAMES = {
    "en": {
        "shogyo": "Commercial (商業)", "kinrin_shogyo": "Neighborhood Commercial (近隣商業)",
        "jun_kogyo": "Quasi-Industrial (準工業)", "dai1_jukyo": "Residential I (第一種住居)",
        "dai2_jukyo": "Residential II (第二種住居)", "jun_jukyo": "Quasi-Residential (準住居)",
        "dai1_chukoso": "Mid/High-Rise Excl. I (第一種中高層)", "dai2_chukoso": "Mid/High-Rise Excl. II (第二種中高層)",
        "dai1_teiso": "Low-Rise Exclusive I (第一種低層)", "dai2_teiso": "Low-Rise Exclusive II (第二種低層)",
        "den_jukyo": "Rural Residential (田園住居)", "kogyo": "Industrial (工業)",
        "kogyo_senyo": "Industrial-Exclusive (工業専用) ⚠", "choseikuiki": "Urbanization Control Area (市街化調整区域) ⚠",
        "unknown": "Unknown / not listed",
    },
    "ja": {
        "shogyo": "商業地域", "kinrin_shogyo": "近隣商業地域",
        "jun_kogyo": "準工業地域", "dai1_jukyo": "第一種住居地域",
        "dai2_jukyo": "第二種住居地域", "jun_jukyo": "準住居地域",
        "dai1_chukoso": "第一種中高層住居専用地域", "dai2_chukoso": "第二種中高層住居専用地域",
        "dai1_teiso": "第一種低層住居専用地域", "dai2_teiso": "第二種低層住居専用地域",
        "den_jukyo": "田園住居地域", "kogyo": "工業地域",
        "kogyo_senyo": "工業専用地域 ⚠", "choseikuiki": "市街化調整区域 ⚠",
        "unknown": "不明・未記載",
    },
}

PTYPE_NAMES = {
    "en": {t: t for t in PROPERTY_TYPES},
    "ja": {
        "1R Mansion": "1Rマンション", "1K Apartment": "1Kアパート",
        "1DK Apartment": "1DKアパート", "1LDK Apartment": "1LDKアパート",
        "2K Apartment": "2Kアパート", "2DK Mansion": "2DKマンション",
        "2LDK Mansion": "2LDKマンション",
        "3DK Mansion": "3DKマンション", "3LDK Mansion": "3LDKマンション",
        "Whole Building Apartment": "一棟アパート",
        "Detached House": "戸建て", "Office Unit": "事務所区分",
        "Corner Retail Unit": "角地店舗区分",
    },
}

TRANSLATIONS = {
    "en": {
        # Sidebar
        "col_title": "Property",
        "col_area": "Area (m²)",
        "col_price": "Price (¥)",
        "col_score": "Score",
        "col_url": "Link",
        "risk_hazard": "Hazard",
        "risk_seismic": "Seismic",
        "risk_liquidity": "Exit liquidity",
        "risk_resale": "Resale outlook",
        "risk_development": "Development",
        "risk_zoning": "Zoning",
        "risk_detail_header": "Risk detail — what each flag means",
        "rv_clear": "no zones hit",
        "rv_unchecked": "not checked",
        "rv_none": "none nearby",
        "rv_low_base": "too few residents",
        "rv_anomaly": "data artifact",
        "rv_new_seismic": "新耐震 · {year}",
        "rv_grey_seismic": "verify · {year}",
        "rv_old_seismic": "旧耐震 · {year}",
        "form_nickname": "Label for this property (optional)",
        "form_nickname_help": "Something you'll recognise later — “Shimogyo 1K by Gojo” beats an auto-generated ID. Shown in the ledger and outcome picker.",
        "statuscheck_btn": "🔄 Check tracked listings",
        "statuscheck_spinner": "Re-visiting each tracked listing…",
        "statuscheck_none": "Nothing to check — every tracked listing already has a recorded outcome.",
        "statuscheck_done": "Checked {checked}: {delisted} delisted, {live} still live. {scored} produced a scored outcome.",
        "statuscheck_unreachable": "{n} listing(s) couldn't be reached — that portal blocks automated requests from this host. Unreachable is not treated as evidence; run this locally, or record those by hand below.",
        "col_nickname": "Property",
        "deep_header": "🔬 Deep dive — one listing, everything recorded",
        "deep_pick": "Listing",
        "deep_ward_chart": "{ward} — median ¥/m² by quarter (MLIT transactions)",
        "deep_no_ward_data": "No quarterly transaction data for {ward} yet — refresh its prefecture in the sidebar.",
        "deep_all_fields": "All recorded fields",
        "rv_to_horizon": "projected",
        "deep_risk_explain": "What each flag means for this deal",
        "deep_hz_line": "🌊 This location sits inside: {zones}. Affects financing, insurance cost and resale — verify against the municipality's official hazard map.",
        "deep_hz_none": "🌊 No flood, landslide, storm-surge or tsunami zone hit at this location.",
        "deep_seismic_ok": "🏚️ Built around {year} — 新耐震 (post-1981 seismic standard). No financing or resale penalty on this count.",
        "deep_resale_none": "📉 No population projection recorded for this listing. Re-analyse it with an MLIT key set to fill this in.",
        "deep_dev_line": "🏗️ Official development designations nearby: {items}. A leading indicator of redevelopment — verify at the ward's 都市計画課.",
        "deep_dev_none": "🏗️ No 高度利用地区, 地区計画 or planned road recorded near this location. Note MLIT's API does not publish 都市再生特別地区 or 特定街区, which drive many large projects.",
        "hzshort_flood": "Flood", "hzshort_landslide": "Landslide",
        "hzshort_surge": "Storm surge", "hzshort_tsunami": "Tsunami",
        "hzshort_rank1": "<0.5m", "hzshort_rank2": "0.5–3m", "hzshort_rank3": "3–5m",
        "hzshort_rank4": "5–10m", "hzshort_rank5": "10–20m", "hzshort_rank6": ">20m",
        "hzshort_class1": "警戒", "hzshort_class2": "特別警戒",
        "devshort_kodo": "高度利用地区", "devshort_road": "Planned road",
        "devshort_chiku": "District plan", "devshort_activity": "Active area",
        "devshort_kodo_near": "Zone <300m", "devshort_chiku_near": "Plan <300m",
        "outcome_clear": "Reset to pending",
        "outcome_cleared": "Cleared the recorded outcome for {listing}.",
        "statuscheck_no_url": "{n} listing(s) skipped: no individual listing URL was saved for them, only a portal homepage. Re-analyse with the listing's URL pasted in to make them checkable.",
        "ward_no_data": "📍 Ward identified as **{ward}**, but no MLIT transaction data has been downloaded for it yet — scoring against the {pref} average meanwhile. Refresh {pref} in the sidebar to fix this.",
        "ward_not_fetched": "📍 Ward identified as **{ward}** — the address resolved fine, but this ward isn't in the fetch list, so no MLIT benchmark exists for it. Scoring against the {pref} average. Ask to have {ward} added to the fetch targets for hyper-local scoring.",
        "vision_provider_down": "The vision provider is temporarily unavailable (HTTP {code}) — this is an outage on their side, not a problem with your image, key or setup. Already retried a few times. Wait a minute and press Extract again, or use the browser extension / type the details in meanwhile.",
        "vision_rate_limited": "Rate limited by the vision provider — too many requests in a short window. Already retried with backoff. Wait a moment and try again.",
        "vision_bad_key": "The vision provider rejected the credentials (HTTP {code}). Check that the API key for your VISION_PROVIDER is set correctly in secrets — retrying won't help with this one.",
        "vision_bad_request": "The vision provider rejected the request (HTTP 400). The image may be too large or in an unsupported format — try a smaller PNG/JPEG screenshot.",
        "source_unknown": "Unknown source",
        "agency_not_identified": "Agency not identified",
        "channel_agent_pdf": "Agent PDF", "channel_agent_image": "Agent photo",
        "channel_portal_screenshot": "Portal screenshot", "channel_listing_url": "Listing URL",
        "channel_browser_extension": "Browser extension", "channel_manual": "Manual entry",
        "doc_empty": "That file appears to be empty.",
        "doc_too_big": "File is {size} MB — the limit is {limit} MB. Export a smaller PDF or downscale the image.",
        "doc_no_pymupdf": "PDF support needs the PyMuPDF package, which isn't installed ({err}). Add `pymupdf` to requirements.txt, or upload page images instead.",
        "doc_corrupt": "That PDF couldn't be opened ({err}). It may be corrupt — try re-exporting it.",
        "doc_encrypted": "That PDF is password-protected. Remove the password and re-upload; the app never asks for document passwords.",
        "doc_page_cap": "Document has {total} pages; only the first {cap} were processed.",
        "doc_conflict_header": "Conflicting values found — please choose",
        "doc_conflict_row": "{field}: keeping “{kept}” (page {page} says “{other}”)",
        "fin_status_not_evaluated": "Not evaluated", "fin_status_acceptable": "Acceptable",
        "fin_status_tight": "Tight", "fin_status_unworkable": "Unworkable",
        "size_header": "📐 Size resilience", 
        "size_caption": "Whether this unit stays functional, rentable, financeable and saleable over a long hold. The area prior reflects today's financing and resale structure — it is NOT a prediction that occupants will want this exact size in 15 years.",
        "size_na": "Size-resilience model applies to sectional residential units only — not applicable to this property type.",
        "size_breakdown": "Final {final} = prior {prior} blended with local evidence {local} at strength {strength} (capped at {cap}). Functional {func} · rental {rental} · exit breadth {exit}. {n} comparables, median similarity {sim}.",
        "size_missing": "Missing evidence (lowers confidence, not score): {items}",
        "rent_contract_label": "Current contractual rent (¥/month)",
        "rent_market_label": "Agent-estimated market rent (¥/month, upside only)",
        "rent_market_help": "Recorded as an upside scenario. It never replaces contractual rent in the base case.",
        "form_building_name": "Building / apartment name",
        "form_unit_number": "Unit no.", "form_unit_floor": "Floor",
        "form_structure": "Structure", "form_occupancy": "Occupancy",
        "form_ingestion_channel": "How this data arrived",
        "form_source_type": "Who supplied it", "form_agency_name": "Agency / brokerage",
        "form_agent_name": "Representative", "form_agency_phone": "Agency phone",
        "form_agency_role": "Agency role", "form_seller_name": "Seller",
        "form_management_company": "Management company",
        "form_rent_guarantee": "Rent-guarantee company",
        "form_source_doc": "Source document",
        "attrib_header": "Optional source and party information",
        "attrib_caption": "All optional, and usually unknown before you speak with the agent. Leave every field blank and the valuation is unaffected — none of it enters the score, fair value, NOI, size resilience or the recommendation. You can fill these in later under Portfolio Audit › Deep dive.",
        "attrib_unknown_entities": "Companies found without a clear role: {items}. Assign them above if you can — they are deliberately NOT auto-filed as the agency.",
        "srctype_real_estate_agency": "Real-estate agency", "srctype_property_portal": "Property portal",
        "srctype_seller": "Seller", "srctype_management_company": "Management company",
        "srctype_user_supplied": "Supplied by me", "srctype_unknown": "Unknown",
        "arole_seller": "Seller", "arole_seller_agent": "Seller's agent",
        "arole_buyer_agent": "Buyer's agent", "arole_intermediary": "Intermediary",
        "arole_document_issuer": "Document issuer", "arole_unknown": "Unknown",
        "doc_conflict_ack": "I've reviewed these — dismiss",
        "size_final": "Size resilience", "size_prior_label": "Area prior",
        "size_evidence_label": "Evidence strength",
        "size_experimental_note": "Evidence-weighted by default: exactly one size mode affects the score. The adjustment is capped and never reaches the weight learner until calibrated by back-testing.",
        "fin_header": "🏦 Optional financing scenario — use after receiving indicative bank terms",
        "fin_caption": "Second-stage analysis. Screening works without any of this, and none of it touches asset quality, fair value or the property score.",
        "fin_down": "Down payment (%)", "fin_rate": "Indicative rate (%/yr)",
        "fin_term": "Loan term (years)", "fin_target_coc": "Target cash-on-cash (%)",
        "fin_status": "Financing", "fin_dscr": "DSCR", "fin_coc": "Cash-on-cash",
        "fin_breakdown": "Cash invested {cash} · annual debt service {debt} · annual cash flow after debt {flow} · maximum offer at your target {maxp}.",
        "fin_isolation_note": "These figures describe the deal's financing, not the building. A strong asset with poor terms is still a strong asset.",
        "sizemode_legacy_penalty": "legacy binary penalty",
        "sizemode_descriptive_only": "descriptive only (no score effect)",
        "sizemode_evidence_weighted": "evidence-weighted",
        "size_mode_note": "Mode: {mode} — this contributed {adj} points to the score (hard-capped at ±{cap}). Not calibrated by back-testing yet, and deliberately kept out of the weight learner.",
        "rent_source_contract": "verified contractual rent", "rent_source_gross_yield": "derived from advertised gross yield",
        "rent_basis_note": "Base case uses {src}. Contractual {c}% vs advertised {a}% ({gap}pp).",
        "rent_conflict_warn": "⚠️ The advertised gross yield implies a rent the contract doesn't support ({gap}pp apart). The base case uses the CONTRACT. Worth asking the agent which figure is current.",
        "rent_unbenchmarked": "Rent is contract-verified but locally unbenchmarked: no local rent median, comparable count, similarity or listing-duration data exists yet, so repeatability can't be judged. Evidence strength is limited accordingly.",
        "form_yield_optional": "Advertised gross yield (%) — optional if rent is entered",
        "form_yield_optional_help": "Leave at 0 when you've entered the contractual rent; the implied yield is calculated from the lease. Fill it in to cross-check the advertised figure against the contract.",
        "form_yield_implied": "Implied gross yield from the contractual rent: {pct}%",
        "col_source": "Source", "col_building": "Building", "col_unit_floor": "Unit / floor",
        "edit_header": "✏️ Add or update source, agency and rent evidence",
        "edit_agency": "Agency / brokerage",
        "edit_agent": "Representative",
        "edit_source": "Source document or URL",
        "edit_manager": "Management company",
        "edit_contract": "Current contractual rent (¥/month)",
        "edit_market": "Market-rent evidence (¥/month)",
        "edit_save": "Save later-stage information",
        "edit_saved": "Later-stage information saved.",
        "homes_url": "Evidence source or document name (optional)",
        "rent_paste_label": "Copied rent evidence text",
        "rent_paste_help": "Paste only the relevant rent table or copied page text. The parser ignores sale prices, AI valuations, deposits, fees and loan repayments.",
        "homes_scan": "Extract rent candidates from pasted text",
        "homes_only": "Paste rent-labelled source text first.",
        "homes_review": "Rent observations for review: ",
        "homes_use_median": "Use reviewed median as market-rent evidence",
        "homes_saved": "Reviewed market-rent evidence saved and size resilience refreshed. Contractual rent was not changed.",
        "yield_label_advertised": "ADVERTISED GROSS YIELD",
        "yield_label_implied": "CONTRACTUAL IMPLIED GROSS YIELD",
        "comp_unavailable_commercial": "Comparable valuation unavailable: no compatible commercial transaction category exists yet. Residential condominium sales trade on entirely different ¥/m² and are deliberately not used to value offices or retail units.",
        "form_station_unknown": "Station distance unknown",
        "form_age_unknown": "Building age unknown",
        "missing_evidence": "Scored with unknown: {items}. These count as neutral — neither credited nor penalised — so the score reflects less information, not a worse property.",
        "homes_header": "Add market-rent evidence manually (optional)",
        "homes_caption": "Paste copied rent evidence from an agency sheet or listing page. The app extracts rent-labelled candidates locally. Tick only genuine comparables; no network fetch is made and no median exists until you review the candidates.",
        "homes_select_prompt": "{n} candidate(s) found. Tick only those comparable to this unit — check the quoted text, layout and area first.",
        "homes_none_selected": "Nothing selected yet, so no median is calculated. Tick the comparable observations above.",
        "homes_summary": "{n} selected · median {median} · IQR spread {disp}%. This is a median of what YOU selected, not a verified local market rate.",
        "homes_mixed_layouts": "You have selected more than one layout ({layouts}). Different layouts rent at different levels, so a single median across them is hard to interpret.",
        "homes_fee_sep": "management fee {fee} listed separately (not added to rent)",
        "homes_area_mismatch": "⚠ area differs materially from this unit",
        "homes_no_observations": "No rent-labelled figures were found. Sale prices, AI valuations, deposits, fees and loan repayments were ignored by design.",
        "homes_unreadable": "The pasted evidence could not be parsed ({err}).",
        "evkind_actual": "actual rent figure",
        "evkind_estimate": "estimate (想定/参考/市場)",
        "paste_header": "Paste full listing text (Select All / Copy)",
        "paste_caption": "For portals that block direct fetching (Rakumachi in particular). Open the listing in your browser, select all, copy, and paste it here. Nothing is requested from the portal. Extracted values only PREFILL the form below — nothing is scored or saved until you press the usual score button.",
        "paste_label": "Pasted listing page text",
        "paste_extract_btn": "🔍 Extract pasted listing",
        "paste_empty": "Nothing pasted yet.",
        "paste_extracted": "Prefilled {n} field(s) from the pasted text. Review them below before scoring.",
        "paste_review_note": "{n} field(s) came from pasted text and are marked as such in provenance. Editing any of them records a manual override.",
        "paste_not_stored": "The pasted page text is held in memory for this extraction only and is not saved to the database.",
        "paste_warn_assumed_rent": "⚠️ This unit is listed as tenanted, but only an ASSUMED rent of ¥{amount}/month was found (想定/表面 figures). That is a projection, not a verified lease. Base NOI will not use it — ask the agent for the 契約賃料 before relying on the yield.",
        "paste_warn_income_mismatch": "⚠️ Annual income ¥{annual} does not match ¥{monthly}/month (which implies ¥{implied}). One of the two figures is wrong — check before scoring.",
        "paste_warn_age_conflict": "⚠️ The page shows 築{shown}年 but the construction date gives {calc} years. The construction date is kept; verify which is current.",
        "paste_warn_missing": "Missing a critical field: {field}. Enter it below before scoring.",
        "edit_reason": "Why is this changing? (optional)",
        "edit_reason_ph": "e.g. agent confirmed the brokerage by email",
        "rev_created": "Revision #{n} recorded — changed: {fields}. The original recommendation was not altered.",
        "rev_no_change": "Nothing changed, so no revision was recorded.",
        "rev_header": "🧾 Original recommendation vs current information",
        "rev_original": "Original recommendation",
        "rev_original_note": "Score {score} on {date}. This is the snapshot the walk-forward learner judges — it is never rewritten by later evidence.",
        "rev_current": "Current revised information",
        "rev_current_note": "{n} revision(s), last changed {at}. These are later findings applied on top of the original record.",
        "rev_score_not_rewritten": "Revisions never overwrite the original score. A refreshed analysis answers a different question — what you would conclude today — and is labelled as such.",
        "rev_history": "Revision history ({n})",
        "console_sub": "Real listings, real benchmarks. Nothing here is simulated.",
        "market_intro": "Ward-level data from MLIT transaction records. Refresh a prefecture in the sidebar to extend coverage.",
        "coverage_header": "Benchmark coverage",
        "coverage_empty": "No ward benchmarks yet. Add your MLIT key in the sidebar and refresh a prefecture — scores fall back to prefecture averages until then.",
        "coverage_note": "{n} of {total} target wards have real transaction data. Wards without it are scored against their prefecture average.",
        "col_ward": "Ward",
        "col_samples": "Transactions",
        "col_updated": "Updated",
        "outcome_header": "📌 Record what happened",
        "outcome_help": "Enter outcomes you actually observed. These are the only results the weight learner uses — an earlier version simulated them randomly, which taught the engine nothing.",
        "outcome_listing": "Listing",
        "outcome_status": "What happened",
        "outcome_days": "Days on market",
        "outcome_save": "Save outcome",
        "outcome_saved": "Recorded outcome for {listing}.",
        "console_title": "🏗️ Property Engine Console",
        "persistence_note": "ℹ️ On Streamlit Community Cloud the SQLite file is not guaranteed to persist — it resets when the app reboots or sleeps. Run locally for permanent walk-forward history.",
        # Tabs
        "tab_market": "📊 Market View",
        "tab_audit": "🧠 Portfolio Audit",
        "tab_weights": "⚙️ KPI Weights",
        # Top deals
        "score_suffix": "Score",
        "price_label": "Asking Price",
        "yield_label": "Gross Yield",
        "station_label": "Station Walk",
        "age_label": "Building Age",
        "minutes_suffix": "min",
        "years_suffix": "yrs",
        "deal_factors": "Factor Breakdown",
        "title_fmt": "{ptype} · {area} m² · {pref} · {station} min walk · {age} yrs",
        "area_label": "Floor Area",
        "sqm_price_label": "Price / m²",
        "col_sqm_price": "¥/m²",
        "wal_warning": "⚠️ SQLite WAL mode unavailable on this filesystem — falling back to default journaling. Concurrent sessions may occasionally see 'database is locked'.",
        # Manual property analyzer
        "tab_analyze": "➕ Analyze Property",
        "analyze_header": "➕ Analyze a Real Listing",
        "analyze_intro": "Found a property on Kenbiya, Rakumachi, or Renosy? Enter its details and the engine scores it with the SAME factor bands and live walk-forward-tuned weights used for scans — so the verdict reflects everything the system has learned.",
        "form_portal": "Portal",
        "form_url": "Listing URL",
        "form_ptype": "Property type",
        "form_pref": "Prefecture",
        "form_price": "Asking price (万円)",
        "form_area": "Floor area (m²)",
        "form_fees": "Monthly fees 管理費+修繕積立金 (¥)",
        "form_fees_help": "The listing's monthly 管理費 + 修繕積立金. Leave 0 and it's estimated from floor area — but the actual figure is what makes net yield trustworthy, so it's worth typing in.",
        "form_yield_basis": "Yield basis (which figure the listing quotes)",
        "form_yield_basis_help": "満室時/想定/表面 assume every unit is let, so a vacancy allowance is deducted. 現況 is the building's CURRENT actual yield and already includes its real occupancy — deducting vacancy again would understate it.",
        "ybasis_full": "満室時 / 想定 / 表面 (full occupancy)",
        "ybasis_current": "現況 (current actual occupancy)",
        "ybasis_unknown": "unknown / not stated",
        "ybasis_warn": "⚠️ Yield basis not stated, so full occupancy is assumed (vacancy deducted). If the listing quotes 現況利回り, switch the basis — otherwise net yield is understated by roughly the vacancy rate.",
        "form_station": "Station walk (min)",
        "form_age": "Building age (yrs)",
        "form_yield": "Gross yield (%)",
        "form_yield_help": "表面利回り = annual rent ÷ asking price × 100. Shown on both Kenbiya and Rakumachi listing pages.",
        "form_track": "Track this property in the Portfolio Audit ledger",
        "analyze_btn": "🧮 Score This Property",
        "invalid_input": "Enter a price and floor area, plus either the contractual rent or an advertised gross yield. Station distance and building age may stay unknown — they score neutrally and are listed as missing evidence.",
        "url_mismatch": "That URL doesn't look like a {portal} link — scoring anyway, but double-check it.",
        "verdict_strong": "STRONG CANDIDATE",
        "verdict_thresh_header": "🎯 Verdict thresholds",
        "verdict_thresh_caption": "Where the labels switch. Real listings mostly score in the 40s–60s, so these decide how selective the app is. Below the lower bar reads SKIP — meaning pass on the deal, not \"passed\".",
        "verdict_strong_label": "STRONG CANDIDATE at or above",
        "verdict_consider_label": "WORTH A LOOK at or above",
        "verdict_consider": "WORTH A LOOK",
        "verdict_pass": "SKIP",
        "vs_benchmark": "vs {pref} benchmark",
        "benchmark_src_mlit": "MLIT ward/city transactions ({n} samples)",
        "benchmark_src_mlit_pref": "MLIT prefecture-wide ({n} samples)",
        "benchmark_src_default": "built-in estimate — connect MLIT for real data",
        "percentile_text": "This score beats {pct:.0f}% of the {n} listings in your latest scan.",
        "tracked_success": "Saved to the Portfolio Audit ledger. Use the tracked-listing status check there to revisit the page: a confirmed delisting is recorded as a sale, and blocked or unreachable pages are not.",
        "manual_audit_note": "📍 Tracked listings are real properties. Status is checked by revisiting the listing page: a confirmed delisting is treated as a sale, and blocked or unreachable pages are never counted as one.",
        "outcome_manual": "📍 Tracked (awaiting live polling)",
        # MLIT benchmarks
        "mlit_section": "🏛️ MLIT Municipality ¥/m² Benchmarks",
        "mlit_key_label": "MLIT API key",
        "mlit_key_help": "Free key from the 不動産情報ライブラリ (reinfolib.mlit.go.jp). Can also be supplied via the MLIT_API_KEY secret or environment variable.",
        "mlit_refresh_btn": "🔄 Refresh benchmarks",
        "mlit_scope_label": "Prefecture to refresh (one per click keeps calls light)",
        "mlit_verify_btn": "🔍 Verify ward codes against MLIT",
        "mlit_verify_ok": "✅ {pref}: all {n} ward codes match MLIT's official 市区町村 list.",
        "mlit_verify_bad": "⚠️ {pref}: {n} code problem(s) — these wards will silently return no data until fixed. {details}",
        "mlit_need_key": "Enter an API key first.",
        "mlit_ok": "Updated — {items}",
        "mlit_err": "Could not update: {items}",
        "mlit_auth_error_label": "API Key Error",
        "mlit_auth_error": "Invalid or expired MLIT API key — refresh aborted. Check the key in settings.",
        "mlit_circuit_label": "Refresh aborted",
        "mlit_circuit_msg": "Several municipalities in a row returned no data — the MLIT API appears unavailable or rate-limiting this session. Aborted early instead of hammering it; the error above shows what the server sent. Try again later.",
        "mlit_note": "Benchmarks are the median ¥/m² of last year's REAL transactions per WARD / CITY (5-digit MLIT 市区町村コード) and drive the Price-per-m² factor — so a central-ward listing is compared to that ward, not a prefecture-wide blend. Unmapped locations fall back to a prefecture estimate. Without a key, built-in estimates are used.",
        "mlit_cities_cached": "Municipality-level data cached for {n} ward(s)/city/cities.",
        "form_city_label": "Municipality",
        "bench_current": "{pref}: ¥{rate:,.0f}/m² · {src}",
        # Urban planning (都市計画) inputs
        "form_zoning": "Zoning / 用途地域",
        "form_address": "Address (for geocoding)",
        "form_address_help": "Full Japanese address from the listing, e.g. 東京都文京区本駒込4-40-6. Used to locate the property; optional, but enables map/hazard features.",
        "address_header": "📍 Address & location",
        "address_tip": "💡 Running locally (VS Code) auto-fills this from the page fetch. Online/mobile: paste or type the address, then tap 📍 Locate.",
        "locate_btn": "📍 Locate",
        "geocode_ok": "🗺️ Located at {lat:.5f}, {lon:.5f}.",
        "geocode_mock": "🗺️ Demo coordinates {lat:.5f}, {lon:.5f} — set GEOCODER='gsi' for real geocoding.",
        "geocode_miss": "🗺️ Couldn't geocode that address — map/hazard features will be unavailable for this listing.",
        "hazard_flood": "flood inundation zone (洪水浸水想定区域)",
        "hazard_landslide": "landslide hazard zone (土砂災害警戒区域)",
        "hazard_surge": "storm-surge inundation zone (高潮浸水想定区域)",
        "net_yield_delta": "net {net}% after costs",
        "net_yield_unknown": "net yield: needs price + gross yield",
        "net_yield_breakdown": "Net yield derivation ({basis}): rent {rent}/yr − {vac}% vacancy − {mgmt}% management − fees {fees}/yr ({src}) − tax {tax}/yr = NOI {noi}/yr → **{net}% net**. Assumptions tunable in KPI Weights.",
        "invested_yield_note": "Acquisition costs {costs} ({pct}% — brokerage 3%+¥60k+tax, 取得税, 登録免許税, 司法書士, 印紙, insurance) → total invested {invested} → **{net_inv}% on capital deployed**. Portals quote yield on price only.",
        "area_sub_loan": "🚪 Exit-liquidity risk: {area}m² is below the ~{floor}m² floor most 投資用ローン require, so resale is largely limited to cash buyers and the bid-ask spread widens. Score reduced by {penalty} points. (Tenant demand for small units is fine — single-person households are growing; this is a FINANCING and RESALE constraint.)",
        "area_caution": "🚪 {area}m² sits under ~{caution}m², where some lenders decline and many Tokyo wards' ワンルーム ordinances set the minimum for new builds. Narrower resale pool; score reduced by {penalty} points.",
        "area_investor": "🚪 Normal investor-sized unit. Below {owner}m² owner-occupiers can't use the 住宅ローン控除, so your resale pool is investors — standard for this asset class.",
        "area_broad": "🚪 At/above {owner}m²: owner-occupiers qualify for the 住宅ローン控除, so the resale pool includes families as well as investors — the widest exit.",
        "atier_sub_loan": "sub-loan-floor", "atier_caution": "lender caution",
        "atier_investor": "investor stock", "atier_broad": "broad exit", "atier_unknown": "unknown",
        "exp_acq_label": "Other acquisition costs (% of price, excl. brokerage)",
        "small_unit_pen_label": "Sub-20m² exit-liquidity penalty (points)",
        "resale_horizon_label": "Resale horizon for population projections (years)",
        "resale_severe": "📉 Resale outlook: the surrounding catchment is projected to lose {pct}% of its population between {frm} and {to} (official 500m-mesh projection). Fewer future buyers AND fewer tenants — the main risk to your exit.",
        "resale_mild": "📉 Resale outlook: catchment population projected {pct}% from {frm} to {to} (official 500m-mesh projection). A mild demographic headwind on resale.",
        "resale_flat": "➡️ Resale outlook: catchment population projected {pct}% from {frm} to {to} — broadly stable (official 500m-mesh projection).",
        "resale_growth": "📈 Resale outlook: catchment population projected {pct}% from {frm} to {to} (official 500m-mesh projection). A growing catchment supports both rent and resale.",
        "resale_penalty_note": "Exit-liquidity penalty increased by {extra} points.",
        "resale_credit_note": "Exit-liquidity penalty reduced by {back} points.",
        "resale_low_base": "📊 Resale outlook: only ~{pop} residents projected in this ~1km catchment (below the {minimum} needed for a percentage to be meaningful) — typical where a tile is mostly river, park, rail or commercial floorspace. No demographic adjustment applied in either direction.",
        "resale_anomaly": "📊 Resale outlook: the projection returned {pct}% between {frm} and {to} (field {field}), which is not physically plausible for a populated catchment. Treated as a data artifact and IGNORED for scoring — not as a real forecast. Worth checking the location against a hazard/population map manually.",
        "resale_unknown": "📊 Population projection unavailable for this location (needs an MLIT key and coverage) — exit-liquidity scored on floor area alone, with no demographic adjustment either way.",
        "fees_actual": "your figure",
        "fees_estimated": "estimated from area",
        "seismic_old": "🏚️ 旧耐震 (pre-1981 seismic standard): built around {year}. Many banks decline or shorten loans on this stock, resale is thinner, earthquake insurance costs more, and 耐震補強 may be needed. Score reduced by {penalty} points. Verify the 検査済証 and ask lenders before committing.",
        "seismic_grey": "⚠️ Built around {year} — almost certainly 新耐震, but buildings completed in 1982–83 can hold a permit issued before 1981-06-01. Worth confirming the building permit date with the seller.",
        "hazard_tsunami": "tsunami inundation zone (津波浸水想定)",
        "hzsev_rank1": "depth <0.5m",
        "hzsev_rank2": "depth 0.5–3m",
        "hzsev_rank3": "depth 3–5m",
        "hzsev_rank4": "depth 5–10m",
        "hzsev_rank5": "depth 10–20m",
        "hzsev_rank6": "depth >20m",
        "hzsev_class1": "warning zone 警戒区域",
        "hzsev_class2": "SPECIAL warning zone 特別警戒区域",
        "hazard_warning": "🌊 HAZARD: this location falls within a {zones}. This is a material risk for income property — it affects financing, insurance cost, and resale. Score penalized by {penalty} points. Verify against the official hazard map before proceeding.",
        "dev_kodo": "high-rise incentive zone (高度利用地区)",
        "dev_road": "planned city road within ~30m (都市計画道路)",
        "dev_kodo_near": "high-rise incentive zone within ~300m (高度利用地区・近接)",
        "dev_activity": "unusually dense development activity in the surrounding ~1km",
        "dev_chiku": "district plan zone (地区計画)",
        "dev_chiku_near": "district plan zone within ~300m (地区計画・近接)",
        "dev_context": "🏗️ Development context: {items}. A leading indicator of redevelopment activity — +{bonus} points applied to the score (tunable in KPI Weights). Verify details at the ward's 都市計画課.",
        "form_zoning_help": "Listed in the 物件概要 table on Kenbiya and Rakumachi. Leave 'Unknown' if not shown — unknown is scored neutrally, never penalized.",
        "form_far": "Designated FAR % (容積率)",
        "form_far_help": "容積率 from the listing's 物件概要 (e.g. 200). Enter 0 if not shown.",
        "zoning_warning": "⚠️ This zoning class is a material risk for income property: rebuilding and bank financing are heavily restricted. Cheap high-yield listings in such areas are a classic trap — verify with the municipality before proceeding.",
        # Screenshot ingestion (vision extraction)
        "ext_header": "🧩 Paste from browser extension",
        "ext_help": "Captured a listing with the local browser extension while logged in? Paste its JSON here. Runs locally; same review-before-scoring as every other path.",
        "ext_paste_label": "Extension JSON payload",
        "ext_ingest_btn": "📥 Import pasted data",
        "ext_ingest_failed": "Couldn't read that payload ({msg}). Make sure it's the JSON the extension copied.",
        "upload_header": "📷 Or upload a listing screenshot",
        "upload_label": "Upload a mobile screenshot (Renosy, Kenbiya, Rakumachi…)",
        "extract_btn": "🔎 Extract listing details from screenshot",
        "extract_spinner": "Reading the screenshot…",
        "extract_success": "Extracted {n} field(s) — review the form below, then score.",
        "extract_failed": "Could not extract from this image: {msg}",
        "warn_pref_unsupported": "Prefecture「{name}」is outside this screener's coverage (Tokyo/Kanagawa/Saitama/Chiba/Osaka/Kyoto/Hyogo/Aichi/Fukuoka) — select the prefecture manually; benchmark-based scores won't be meaningful for it.",
        "warn_address_mismatch": "Ignored an address in {addr_pref} that doesn't match the listing's prefecture ({listing_pref}) — it looked like a page footer/company address, not the property. Enter the address manually if needed.",
        "warn_layout_unknown": "Could not map layout「{name}」to a property type — please pick one manually.",
        "warn_yield_computed": "Yield wasn't shown; computed {gy:.2f}% from rent ¥{rent:,.0f}/mo ÷ price.",
        "warn_mock_provider": "Vision provider is 'mock' — these are CANNED demo values, not your screenshot. Set VISION_PROVIDER (openai / gemini / anthropic) and the matching API key to extract for real.",
        "vision_note": "Extraction uses a Vision LLM (set VISION_PROVIDER + API key via secrets/env). Extracted values only pre-fill the form — you stay in control and can correct anything before scoring.",
        # URL import
        "url_import_header": "🔗 Paste a listing URL",
        "url_autofill_info": "Auto-filled from the URL itself: {items}. The rest still needs the page, a screenshot, or your input.",
        "url_fetch_btn": "🌐 Fetch & extract from page",
        "url_fetch_spinner": "Fetching the listing page…",
        "url_fetch_note": "One user-initiated fetch of the page you pasted — not bulk scraping; you remain responsible for the portal's terms of use. Title parsing (price/yield) needs no API key; full extraction uses your configured LLM provider. If the page is login-walled (common on Rakumachi) or blocks the request, use the screenshot path instead.",
        "url_fetch_failed": "Couldn't read that page ({msg}). Try the screenshot upload instead — it always works.",
        "url_fetch_blocked": "This portal blocked the automated fetch (common on Rakumachi — it's their access policy, not an app error). Use the 📷 screenshot upload below instead; it captures from your own browser and is never blocked.",
        "url_fetch_blocked_partial": "This portal blocked the automated fetch (common on Rakumachi). Filled in {items} from the URL itself — add price, area, yield etc. via the 📷 screenshot upload below.",
        "ward_resolved": "📍 Scoring against **{ward}** (ward-level MLIT benchmark).",
        "ward_fallback": "📍 No ward-level benchmark for this address — scoring against the **{pref}** prefecture average. (Tip: a full address incl. the ward, e.g. 渋谷区, enables hyper-local scoring.)",
        "ward_trend": "📈 {ward} price trend: {pct:+.1f}%/yr (from {q} quarters, {n} MLIT samples). Display-only — not yet part of the score.",
        "momentum_header": "📈 Ward momentum ranking",
        "momentum_caption": "Annualized ¥/m² trend from MLIT quarterly medians — {n} wards with enough data. Informational only; refresh benchmarks to extend coverage.",
        "momentum_axis": "annualized price trend (%/yr)",
        # Market view
        "market_header": "📊 Full Market Scan Results",
        # Columns
        "col_prefecture": "Prefecture",
        "col_rec_date": "Recommended",
        "col_orig_price": "Original Price (¥)",
        "col_current_status": "Current Status",
        "col_days_listed": "Days Listed",
        "col_outcome": "Outcome",
        "col_factor": "Factor",
        "col_weight": "Weight",
        "col_change": "Δ vs Default",
        # Statuses / outcomes
        "status_active": "Active",
        "status_sold": "SOLD / Delisted",
        "status_price_drop": "Price Reduced",
        "status_unchanged": "Still Active",
        "outcome_win": "✅ Fast sale — engine validated",
        "outcome_loss": "❌ Stale / discounted — engine penalized",
        "outcome_neutral": "➖ No signal yet",
        "outcome_pending": "⏳ Awaiting reality check",
        # Audit tab
        "audit_header": "🧠 Portfolio Audit — Walk-Forward Tracking",
        "audit_intro": "Every tracked listing is archived here with its exact URL, price and date. The status check revisits each one: a confirmed delisting is treated as a sale (a fast one counts as a good call, a later one is neutral), while blocked or unreachable pages are not delistings. Outcomes you record by hand are the learner's source of truth.",
        "audit_empty": "Nothing tracked yet. Analyse a listing and tick “Track this property” to start building your ledger.",
        "audit_table_header": "Historical Picks Ledger",
        "learning_header": "🤖 Live System Weights (the engine learning)",
        "learning_caption": "Fast sales (< {fast} days) nudge a pick's dominant factors UP; price cuts or {stale}+ days on market nudge them DOWN. Weights are floored at {floor:.0%} and renormalized to sum to 1.",
        "weight_history_header": "Weight Evolution",
        "accuracy_label": "Validated Picks",
        "checked_label": "Picks Evaluated",
        "winrate_label": "Fast-Sale Rate",
        # Weights tab
        "weights_header": "⚙️ KPI Weights Customizer",
        "gates_header": "🚦 Location gates (hazard & development)",
        "gates_caption": "These act on the FINAL score, outside the weighted factors: hazard zones subtract the penalty scaled by the WORST zone's severity (flood depth rank 0.5x–1.5x, landslide 警戒 0.8x / 特別警戒 1.3x; storm surge & tsunami at 1.0x); development signals add a bonus. Strong signals (inside a 高度利用地区 / 地区計画 zone, or a planned road at the parcel) count as full flags; contextual ones (zone within ~300m / unusually active surrounding km) count as half flags. Capped at 2 full flags. Honest gap: MLIT's API does not expose 都市再生特別地区 or 特定街区 designations, which drive many of Japan's largest redevelopments (e.g. Shibuya Scramble Square, Osaka's Brillia Tower Dojima) — those can show no flag here even though real redevelopment is happening; verify at the ward's 都市計画課 for major towers. The 'active area' threshold self-calibrates to 2x the median of tiles you've actually analyzed (once 10+ observed).",
        "hazard_pen_label": "Hazard penalty (points, applied once if any zone hits)",
        "seismic_pen_label": "旧耐震 penalty (points, pre-1981 buildings)",
        "expenses_header": "💰 Net-yield expense assumptions",
        "expenses_caption": "These convert advertised gross yield (満室時利回り, before any cost) into net NOI yield — the figure that actually determines whether a deal works. A listing's real 管理費+修繕積立金 always overrides the per-m² estimate when you enter it.",
        "exp_fee_label": "管理費+修繕積立金 estimate (¥/m²/month)",
        "exp_owner_label": "一棟/戸建 owner maintenance (% of rent)",
        "exp_vacancy_label": "Assumed vacancy (%)",
        "exp_mgmt_label": "Rental management fee (% of rent)",
        "exp_tax_label": "固定資産税+都市計画税 (annual, % of price)",
        "dev_bonus_label": "Development bonus (points per flag, max 2 flags)",
        "weights_intro": "These baseline weights drive the 100-point property score. Manual changes are saved to the same `system_weights` ledger the walk-forward loop writes to — the engine continues learning from your starting point.",
        "weight_slider_help": "Relative importance — all weights are renormalized to sum to 1.0 on save.",
        "save_weights_btn": "💾 Save Weights",
        "reset_weights_btn": "↩️ Reset to Defaults",
        "weights_saved": "Weights saved and renormalized.",
        "weights_reset": "Weights reset to factory defaults.",
        "current_weights_header": "Active Weights",
        "note_manual": "manual adjustment",
        "note_reset": "reset to defaults",
        "note_initial": "initial defaults",
        "note_walkforward": "Walk-forward update on {n} reality-checked pick(s)",
        "formula_header": "Scoring Formula",
        "formula_body": "Score = 100 × (w_yield × YieldScore + w_station × StationScore + w_age × AgeScore + w_sqm × PriceEfficiencyScore + w_dev × DevelopmentScore). Yield is scored against a 4–12% band; station walk against 1–20 min (closer = better); age against 0–45 yrs (newer = better, pre-1981 penalty beyond 45); price efficiency against the listing's ¥/m² vs its prefecture benchmark in a 0.65×–1.35× band (cheaper than market = better); development potential blends the 用途地域 zoning class (commercial flexible = high; 市街化調整区域 / 工業専用 = near zero) with designated 容積率 FAR on a 100–500% band. Unknown zoning scores a neutral 50.",
    },
    "ja": {
        # Sidebar
        "col_title": "物件",
        "col_area": "面積（㎡）",
        "col_price": "価格（円）",
        "col_score": "スコア",
        "col_url": "リンク",
        "risk_hazard": "ハザード",
        "risk_seismic": "耐震",
        "risk_liquidity": "出口流動性",
        "risk_resale": "売却見通し",
        "risk_development": "開発",
        "risk_zoning": "用途地域",
        "risk_detail_header": "リスク詳細 — 各フラグの意味",
        "rv_clear": "該当なし",
        "rv_unchecked": "未確認",
        "rv_none": "周辺になし",
        "rv_low_base": "人口が僅少",
        "rv_anomaly": "データ異常",
        "rv_new_seismic": "新耐震 · {year}年",
        "rv_grey_seismic": "要確認 · {year}年",
        "rv_old_seismic": "旧耐震 · {year}年",
        "form_nickname": "この物件のラベル（任意）",
        "form_nickname_help": "後から見て分かる名前を。「五条 下京区 1K」のように。台帳と結果入力の一覧に表示されます。",
        "statuscheck_btn": "🔄 追跡中の物件を確認",
        "statuscheck_spinner": "各物件のページを再確認しています…",
        "statuscheck_none": "確認対象がありません — 追跡中の物件はすべて結果が記録済みです。",
        "statuscheck_done": "{checked}件を確認: 掲載終了{delisted}件、掲載中{live}件。うち{scored}件を勝敗として記録しました。",
        "statuscheck_unreachable": "{n}件にアクセスできませんでした — このホストからの自動リクエストを遮断するポータルです。アクセス不可は判断材料にしていません。ローカル環境で実行するか、下の手動入力をご利用ください。",
        "col_nickname": "物件",
        "deep_header": "🔬 詳細分析 — 1物件の記録すべて",
        "deep_pick": "物件",
        "deep_ward_chart": "{ward} — 四半期別 ¥/㎡ 中央値（MLIT取引データ）",
        "deep_no_ward_data": "{ward} の四半期データがまだありません — サイドバーで該当都道府県を更新してください。",
        "deep_all_fields": "記録された全項目",
        "rv_to_horizon": "予測",
        "deep_risk_explain": "各フラグがこの物件にとって何を意味するか",
        "deep_hz_line": "🌊 この所在地は次の区域内です: {zones}。融資・保険料・売却に影響します。自治体の公式ハザードマップで必ずご確認ください。",
        "deep_hz_none": "🌊 洪水・土砂災害・高潮・津波のいずれの区域にも該当しません。",
        "deep_seismic_ok": "🏚️ {year}年頃築 — 新耐震基準です。この点での融資・売却上の減点はありません。",
        "deep_resale_none": "📉 この物件には将来推計人口が記録されていません。MLITキーを設定して再分析すると取得できます。",
        "deep_dev_line": "🏗️ 周辺の公式な開発指定: {items}。再開発の先行指標です。各区の都市計画課でご確認ください。",
        "deep_dev_none": "🏗️ 周辺に高度利用地区・地区計画・都市計画道路の記録はありません。なお、MLITのAPIは大規模再開発を規定する都市再生特別地区・特定街区を提供していません。",
        "hzshort_flood": "洪水", "hzshort_landslide": "土砂",
        "hzshort_surge": "高潮", "hzshort_tsunami": "津波",
        "hzshort_rank1": "0.5m未満", "hzshort_rank2": "0.5〜3m", "hzshort_rank3": "3〜5m",
        "hzshort_rank4": "5〜10m", "hzshort_rank5": "10〜20m", "hzshort_rank6": "20m超",
        "hzshort_class1": "警戒", "hzshort_class2": "特別警戒",
        "devshort_kodo": "高度利用地区", "devshort_road": "都市計画道路",
        "devshort_chiku": "地区計画", "devshort_activity": "開発密度高",
        "devshort_kodo_near": "区域300m内", "devshort_chiku_near": "地区計画300m内",
        "outcome_clear": "未確定に戻す",
        "outcome_cleared": "{listing} の記録済み結果を取り消しました。",
        "statuscheck_no_url": "{n}件をスキップしました: 個別物件のURLが保存されておらず、ポータルのトップページのみが記録されています。物件URLを貼り付けて再分析すると確認対象になります。",
        "ward_no_data": "📍 市区町村は **{ward}** と判定されましたが、MLIT取引データが未取得です。暫定的に{pref}平均でスコアリングしています。サイドバーで{pref}を更新してください。",
        "ward_not_fetched": "📍 市区町村は **{ward}** と判定されました（住所の解析は成功）。ただしこの市区町村は取得対象リストに含まれていないため、ベンチマークが存在しません。{pref}平均でスコアリングしています。{ward} を取得対象に追加すると、より局所的なスコアリングが可能になります。",
        "vision_provider_down": "ビジョンプロバイダが一時的に利用できません（HTTP {code}）。画像・APIキー・設定の問題ではなく、提供元側の障害です。すでに数回再試行しています。少し待って再度「抽出」を押すか、ブラウザ拡張または手入力をご利用ください。",
        "vision_rate_limited": "ビジョンプロバイダのレート制限に達しました（短時間にリクエストが集中）。バックオフ付きで再試行済みです。少し待ってから再度お試しください。",
        "vision_bad_key": "ビジョンプロバイダが認証情報を拒否しました（HTTP {code}）。VISION_PROVIDER に対応するAPIキーがsecretsに正しく設定されているか確認してください。再試行では解決しません。",
        "vision_bad_request": "ビジョンプロバイダがリクエストを拒否しました（HTTP 400）。画像が大きすぎるか、対応していない形式の可能性があります。小さめのPNG/JPEGでお試しください。",
        "source_unknown": "提供元不明",
        "agency_not_identified": "仲介会社は特定されていません",
        "channel_agent_pdf": "業者PDF", "channel_agent_image": "業者提供画像",
        "channel_portal_screenshot": "ポータルのスクリーンショット", "channel_listing_url": "物件URL",
        "channel_browser_extension": "ブラウザ拡張", "channel_manual": "手入力",
        "doc_empty": "ファイルが空のようです。",
        "doc_too_big": "ファイルサイズが{size}MBです（上限{limit}MB）。PDFを軽くするか、画像を縮小してください。",
        "doc_no_pymupdf": "PDF処理にはPyMuPDFが必要ですが、インストールされていません（{err}）。requirements.txtに`pymupdf`を追加するか、ページ画像をアップロードしてください。",
        "doc_corrupt": "PDFを開けませんでした（{err}）。破損している可能性があります。書き出し直してお試しください。",
        "doc_encrypted": "このPDFはパスワードで保護されています。解除してから再アップロードしてください（本アプリがパスワードを尋ねることはありません）。",
        "doc_page_cap": "全{total}ページのうち、最初の{cap}ページのみ処理しました。",
        "doc_conflict_header": "値が食い違っています — ご確認ください",
        "doc_conflict_row": "{field}: 「{kept}」を採用（{page}ページには「{other}」）",
        "fin_status_not_evaluated": "未評価", "fin_status_acceptable": "問題なし",
        "fin_status_tight": "ぎりぎり", "fin_status_unworkable": "成立困難",
        "size_header": "📐 サイズ耐性",
        "size_caption": "長期保有において、この住戸が機能的で・貸せて・融資が付き・売れる状態を保てるかの評価です。面積の基準値は現在の融資・売却構造を反映したものであり、15年後の入居者が同じ広さを求めるという予測ではありません。",
        "size_na": "サイズ耐性モデルは区分住戸のみが対象です。この物件種別には適用されません。",
        "size_breakdown": "最終{final} = 基準値{prior} と 現地エビデンス{local} を強度{strength}で加重（上限{cap}）。機能性{func}・賃貸耐性{rental}・出口の広さ{exit}。比較事例{n}件、類似度中央値{sim}。",
        "size_missing": "不足している情報（スコアではなく確信度を下げます）: {items}",
        "rent_contract_label": "現行の契約賃料（円/月）",
        "rent_market_label": "業者想定の市場賃料（円/月・アップサイドのみ）",
        "rent_market_help": "アップサイドのシナリオとして記録します。基本ケースの契約賃料を置き換えることはありません。",
        "form_building_name": "建物・マンション名",
        "form_unit_number": "部屋番号", "form_unit_floor": "所在階",
        "form_structure": "構造", "form_occupancy": "稼働状況",
        "form_ingestion_channel": "データの入手経路",
        "form_source_type": "情報の提供元", "form_agency_name": "仲介会社",
        "form_agent_name": "担当者", "form_agency_phone": "仲介会社 電話番号",
        "form_agency_role": "仲介会社の立場", "form_seller_name": "売主",
        "form_management_company": "管理会社",
        "form_rent_guarantee": "家賃保証会社",
        "form_source_doc": "元資料",
        "attrib_header": "任意：情報源・関係者情報",
        "attrib_caption": "すべて任意です。仲介会社に問い合わせる前は不明なことが多いため、空欄のままで問題ありません。スコア・適正価格・NOI・サイズ耐性・推奨判定のいずれにも影響しません。後から「ポートフォリオ監査 › 詳細分析」で追加できます。",
        "attrib_unknown_entities": "役割が不明な会社: {items}。可能であれば上で割り当ててください（自動で仲介会社として登録することは意図的に行いません）。",
        "srctype_real_estate_agency": "不動産会社", "srctype_property_portal": "物件ポータル",
        "srctype_seller": "売主", "srctype_management_company": "管理会社",
        "srctype_user_supplied": "自分で入力", "srctype_unknown": "不明",
        "arole_seller": "売主", "arole_seller_agent": "売主側仲介",
        "arole_buyer_agent": "買主側仲介", "arole_intermediary": "仲介",
        "arole_document_issuer": "資料発行元", "arole_unknown": "不明",
        "doc_conflict_ack": "確認しました — 閉じる",
        "size_final": "サイズ耐性", "size_prior_label": "面積基準値",
        "size_evidence_label": "エビデンス強度",
        "size_experimental_note": "既定はエビデンス加重です。スコアには常に1つのサイズモードだけが作用し、調整幅は上限付きです。バックテストで較正されるまでウェイト学習には渡しません。",
        "fin_header": "🏦 任意の融資シナリオ（金融機関の条件提示後に使用）",
        "fin_caption": "第2段階の分析です。初期スクリーニングはこれらの入力なしで完結し、資産の質・適正価格・スコアには一切影響しません。",
        "fin_down": "自己資金比率（%）", "fin_rate": "想定金利（年%）",
        "fin_term": "借入期間（年）", "fin_target_coc": "目標自己資金利回り（%）",
        "fin_status": "融資判定", "fin_dscr": "DSCR", "fin_coc": "自己資金利回り",
        "fin_breakdown": "自己資金 {cash}・年間返済額 {debt}・返済後キャッシュフロー {flow}・目標達成に必要な上限価格 {maxp}。",
        "fin_isolation_note": "これらは融資条件の話であり、建物の評価ではありません。条件が悪くても、良い物件は良い物件です。",
        "sizemode_legacy_penalty": "従来の一律減点",
        "sizemode_descriptive_only": "参考表示のみ（スコア非反映）",
        "sizemode_evidence_weighted": "エビデンス加重",
        "size_mode_note": "モード: {mode} — スコアへの寄与は{adj}点（上限±{cap}点）。バックテストによる較正は未実施で、ウェイト学習には意図的に渡していません。",
        "rent_source_contract": "確認済みの契約賃料", "rent_source_gross_yield": "表示利回りからの逆算",
        "rent_basis_note": "基本ケースは{src}を使用。契約ベース{c}% / 表示{a}%（差{gap}pp）。",
        "rent_conflict_warn": "⚠️ 表示利回りが示す賃料と契約賃料が{gap}pp乖離しています。基本ケースは契約賃料を採用しています。どちらが現行か仲介会社にご確認ください。",
        "rent_unbenchmarked": "賃料は契約で確認済みですが、周辺相場との比較ができていません（地域の賃料中央値・比較件数・類似度・募集期間のデータが未取得）。再現性を判断できないため、エビデンス強度は限定的に扱っています。",
        "form_yield_optional": "表示利回り（%）— 賃料入力時は任意",
        "form_yield_optional_help": "契約賃料を入力している場合は0のままで構いません（賃料から利回りを計算します）。入力すると、表示利回りと契約賃料の整合性を確認できます。",
        "form_yield_implied": "契約賃料から算出した利回り: {pct}%",
        "col_source": "情報源", "col_building": "建物", "col_unit_floor": "部屋/階",
        "edit_header": "✏️ 情報源・仲介会社・賃料エビデンスの追加/更新",
        "edit_agency": "仲介会社",
        "edit_agent": "担当者",
        "edit_source": "元資料またはURL",
        "edit_manager": "管理会社",
        "edit_contract": "現行の契約賃料（円/月）",
        "edit_market": "市場賃料エビデンス（円/月）",
        "edit_save": "後から得た情報を保存",
        "edit_saved": "後から得た情報を保存しました。",
        "homes_url": "情報源または資料名（任意）",
        "rent_paste_label": "コピーした賃料エビデンス",
        "rent_paste_help": "賃料表またはページのコピーを貼り付けてください。販売価格・AI査定・敷金・費用・返済額は除外します。",
        "homes_scan": "貼り付けテキストから賃料候補を抽出",
        "homes_only": "賃料ラベルのある元テキストを貼り付けてください。",
        "homes_review": "確認用の賃料実績: ",
        "homes_use_median": "確認済みの中央値を市場賃料エビデンスとして使用",
        "homes_saved": "確認済み市場賃料エビデンスを保存し、サイズ耐性を更新しました。契約賃料は変更していません。",
        "yield_label_advertised": "表示利回り",
        "yield_label_implied": "契約賃料ベースの利回り",
        "comp_unavailable_commercial": "比較評価は利用できません: 対応する商業用の取引区分がまだありません。住居用マンションの成約事例は㎡単価の水準が全く異なるため、事務所・店舗の評価には意図的に使用していません。",
        "form_station_unknown": "駅徒歩は不明",
        "form_age_unknown": "築年数は不明",
        "missing_evidence": "不明のまま評価した項目: {items}。中立として扱っています（加点も減点もしません）。スコアは物件が劣ることではなく、情報が少ないことを反映しています。",
        "homes_header": "市場賃料エビデンスを手動追加（任意）",
        "homes_caption": "仲介資料や物件ページからコピーした賃料情報を貼り付けてください。賃料ラベル付き候補をローカルで抽出します。比較可能なものだけを選択するまで中央値は算出せず、ネット取得も行いません。",
        "homes_select_prompt": "{n}件の候補が見つかりました。引用テキスト・間取り・面積を確認し、この住戸と比較可能なものだけにチェックを入れてください。",
        "homes_none_selected": "まだ選択されていないため、中央値は算出していません。上で比較可能な実績を選択してください。",
        "homes_summary": "{n}件を選択 · 中央値 {median} · 四分位範囲 {disp}%。これは「ご自身が選択した」実績の中央値であり、検証済みの地域相場ではありません。",
        "homes_mixed_layouts": "複数の間取り（{layouts}）が選択されています。間取りが異なると賃料水準も異なるため、単一の中央値は解釈が難しくなります。",
        "homes_fee_sep": "管理費 {fee} は別掲（賃料には加算していません）",
        "homes_area_mismatch": "⚠ 面積が本住戸と大きく異なります",
        "homes_no_observations": "賃料ラベル付き金額は見つかりませんでした。販売価格・AI査定・敷金・費用・返済額は仕様により除外しています。",
        "homes_unreadable": "貼り付けた情報を解析できませんでした（{err}）。",
        "evkind_actual": "実賃料",
        "evkind_estimate": "想定・参考・市場賃料",
        "paste_header": "物件ページ全文を貼り付け（すべて選択 / コピー）",
        "paste_caption": "直接取得がブロックされるポータル（特に楽待）向けです。ブラウザで物件ページを開き、すべて選択してコピーし、ここに貼り付けてください。ポータルへのリクエストは一切行いません。抽出値は下のフォームに自動入力されるだけで、通常のスコアボタンを押すまで採点も保存も行いません。",
        "paste_label": "貼り付けた物件ページのテキスト",
        "paste_extract_btn": "🔍 貼り付けた内容から抽出",
        "paste_empty": "まだ何も貼り付けられていません。",
        "paste_extracted": "{n}項目を貼り付けテキストから入力しました。採点前にご確認ください。",
        "paste_review_note": "{n}項目が貼り付けテキスト由来です（出所情報に記録済み）。編集すると手動上書きとして記録されます。",
        "paste_not_stored": "貼り付けたページ本文は抽出処理中のみメモリ上に保持し、データベースには保存しません。",
        "paste_warn_assumed_rent": "⚠️ 賃貸中と表示されていますが、見つかったのは想定賃料 月額{amount}円のみです。これは想定値であり、確認済みの契約賃料ではありません。基本ケースのNOIには使用しません。利回りを判断する前に契約賃料を仲介会社にご確認ください。",
        "paste_warn_income_mismatch": "⚠️ 年間収入{annual}円と月額{monthly}円（年換算{implied}円）が一致しません。いずれかが誤りです。採点前にご確認ください。",
        "paste_warn_age_conflict": "⚠️ ページ上の築{shown}年に対し、築年月からの計算では{calc}年です。築年月を採用していますが、どちらが正しいかご確認ください。",
        "paste_warn_missing": "重要項目が不足しています: {field}。採点前に下で入力してください。",
        "edit_reason": "変更理由（任意）",
        "edit_reason_ph": "例: 仲介会社をメールで確認",
        "rev_created": "改訂#{n}を記録しました — 変更項目: {fields}。当初の推奨内容は変更していません。",
        "rev_no_change": "変更がないため、改訂は記録されませんでした。",
        "rev_header": "🧾 当初の推奨 と 現在の情報",
        "rev_original": "当初の推奨",
        "rev_original_note": "{date} 時点のスコア {score}。ウォークフォワード学習が評価するのはこのスナップショットであり、後から得た情報で書き換えられることはありません。",
        "rev_current": "現在の改訂情報",
        "rev_current_note": "改訂{n}件、最終更新 {at}。当初の記録の上に後から判明した情報を適用したものです。",
        "rev_score_not_rewritten": "改訂が当初スコアを上書きすることはありません。再評価は「今なら何と判断するか」という別の問いであり、その旨を明示して表示します。",
        "rev_history": "改訂履歴（{n}件）",
        "console_sub": "実際の物件と実データのベンチマークのみ。シミュレーションは含みません。",
        "market_intro": "MLIT取引データに基づく市区町村レベルの情報です。サイドバーで都道府県を更新すると対象が広がります。",
        "coverage_header": "ベンチマーク網羅状況",
        "coverage_empty": "市区町村ベンチマークがまだありません。サイドバーでMLITキーを設定し、都道府県を更新してください。それまでは都道府県平均でスコアリングされます。",
        "coverage_note": "対象{total}市区町村のうち{n}件に実取引データがあります。データのない市区町村は都道府県平均で評価されます。",
        "col_ward": "市区町村",
        "col_samples": "取引件数",
        "col_updated": "更新日",
        "outcome_header": "📌 実際の結果を記録",
        "outcome_help": "実際に確認できた結果のみを入力してください。ウェイト学習が使うのはこのデータだけです（以前のバージョンは乱数で結果を模擬しており、学習の意味がありませんでした）。",
        "outcome_listing": "物件",
        "outcome_status": "結果",
        "outcome_days": "売出日数",
        "outcome_save": "結果を保存",
        "outcome_saved": "{listing} の結果を記録しました。",
        "console_title": "🏗️ 物件エンジン・コンソール",
        "persistence_note": "ℹ️ Streamlit Community CloudではSQLiteファイルの永続化は保証されません — アプリの再起動やスリープでリセットされます。ウォークフォワード履歴を恒久的に残すにはローカルで実行してください。",
        # Tabs
        "tab_market": "📊 マーケットビュー",
        "tab_audit": "🧠 ポートフォリオ監査",
        "tab_weights": "⚙️ KPIウェイト",
        # Top deals
        "score_suffix": "スコア",
        "price_label": "販売価格",
        "yield_label": "表面利回り",
        "station_label": "駅徒歩",
        "age_label": "築年数",
        "minutes_suffix": "分",
        "years_suffix": "年",
        "deal_factors": "ファクター内訳",
        "title_fmt": "{ptype}・{area}㎡・{pref}・徒歩{station}分・築{age}年",
        "area_label": "専有面積",
        "sqm_price_label": "㎡単価",
        "col_sqm_price": "円/㎡",
        "wal_warning": "⚠️ このファイルシステムではSQLiteのWALモードが利用できません — 標準ジャーナリングで動作します。同時セッションで「database is locked」が発生する場合があります。",
        # Manual property analyzer
        "tab_analyze": "➕ 物件を診断",
        "analyze_header": "➕ 実在物件を診断",
        "analyze_intro": "健美家・楽待・Renosyで見つけた物件の情報を入力すると、スキャンと同一のファクター基準と、ウォークフォワード学習済みの最新ウェイトでスコアリングします — システムが学習した内容がそのまま判定に反映されます。",
        "form_portal": "ポータル",
        "form_url": "物件URL",
        "form_ptype": "物件種別",
        "form_pref": "都道府県",
        "form_price": "販売価格（万円）",
        "form_area": "専有面積（㎡）",
        "form_fees": "月額 管理費+修繕積立金（円）",
        "form_fees_help": "物件の月額管理費＋修繕積立金。0のままなら専有面積から推定しますが、実額を入力するとネット利回りの精度が大きく上がります。",
        "form_yield_basis": "利回りの基準（物件がどの利回りを表示しているか）",
        "form_yield_basis_help": "満室時・想定・表面利回りは全室稼働を前提とするため、空室率を控除します。現況利回りは現在の稼働状況を既に反映しているため、重ねて空室控除すると過小評価になります。",
        "ybasis_full": "満室時・想定・表面（全室稼働前提）",
        "ybasis_current": "現況（現在の稼働状況）",
        "ybasis_unknown": "不明・記載なし",
        "ybasis_warn": "⚠️ 利回りの基準が不明のため、全室稼働前提として空室率を控除しています。物件が現況利回りを表示している場合は基準を変更してください（そのままでは空室率相当分だけ過小評価になります）。",
        "form_station": "駅徒歩（分）",
        "form_age": "築年数（年）",
        "form_yield": "表面利回り（%）",
        "form_yield_help": "表面利回り = 年間賃料 ÷ 販売価格 × 100。健美家・楽待の物件ページに表示されています。",
        "form_track": "この物件をポートフォリオ監査台帳で追跡する",
        "analyze_btn": "🧮 この物件をスコアリング",
        "invalid_input": "価格と専有面積、そして契約賃料または表示利回りのいずれかを入力してください。駅徒歩・築年数は不明のままでも構いません（中立評価となり、不足情報として表示されます）。",
        "url_mismatch": "このURLは{portal}のリンクではない可能性があります — スコアリングは実行しますが、ご確認ください。",
        "verdict_strong": "有力候補",
        "verdict_thresh_header": "🎯 判定のしきい値",
        "verdict_thresh_caption": "ラベルが切り替わる基準です。実際の物件は40〜60点台に集中するため、この設定がアプリの厳しさを決めます。下限未満は「見送り」です。",
        "verdict_strong_label": "「有力候補」とする下限",
        "verdict_consider_label": "「要検討」とする下限",
        "verdict_consider": "要検討",
        "verdict_pass": "見送り",
        "vs_benchmark": "{pref}基準比",
        "benchmark_src_mlit": "国交省 成約データ（市区町村 {n}件）",
        "benchmark_src_mlit_pref": "国交省 都道府県全体（{n}件）",
        "benchmark_src_default": "内蔵推定値 — 国交省APIに接続すると実データになります",
        "percentile_text": "このスコアは直近スキャンの{n}件中、{pct:.0f}%の物件を上回ります。",
        "tracked_success": "ポートフォリオ監査台帳に保存しました。台帳のステータス確認から物件ページを再訪できます。掲載終了が確認できた場合は売却として記録し、アクセス不可・ブロックされたページは対象外です。",
        "manual_audit_note": "📍 追跡中の物件はすべて実在の物件です。ステータスは物件ページを再訪して確認します。掲載終了が確認できた場合は売却とみなし、アクセス不可・ブロックされたページは掲載終了として扱いません。",
        "outcome_manual": "📍 追跡中（ライブ取得待ち）",
        # MLIT benchmarks
        "mlit_section": "🏛️ 国交省 市区町村㎡単価ベンチマーク",
        "mlit_key_label": "国交省 APIキー",
        "mlit_key_help": "不動産情報ライブラリ（reinfolib.mlit.go.jp）で無料発行。MLIT_API_KEY のシークレット/環境変数でも設定できます。",
        "mlit_refresh_btn": "🔄 ベンチマークを更新",
        "mlit_scope_label": "更新する都道府県（1回につき1都道府県で負荷を軽減）",
        "mlit_verify_btn": "🔍 市区町村コードをMLITと照合",
        "mlit_verify_ok": "✅ {pref}: {n}件の市区町村コードすべてがMLITの公式一覧と一致しました。",
        "mlit_verify_bad": "⚠️ {pref}: {n}件のコードに問題があります — 修正するまで該当市区町村はデータを取得できません。{details}",
        "mlit_need_key": "先にAPIキーを入力してください。",
        "mlit_ok": "更新しました — {items}",
        "mlit_err": "更新できませんでした: {items}",
        "mlit_auth_error_label": "APIキーエラー",
        "mlit_auth_error": "MLIT APIキーが無効または期限切れです — 更新を中止しました。設定を確認してください。",
        "mlit_circuit_label": "更新を中止",
        "mlit_circuit_msg": "複数の市区町村で連続してデータが取得できませんでした — MLIT APIが利用不可またはレート制限中の可能性があります。過剰なリクエストを避けるため早期中止しました。上記のエラーにサーバーの応答内容が表示されています。時間をおいて再試行してください。",
        "mlit_note": "ベンチマークは市区町村ごと（5桁の国交省 市区町村コード）の昨年の実成約価格の中央値㎡単価で、㎡単価ファクターの基準になります — 中心区の物件は都道府県全体の平均ではなく、その区の実データと比較されます。未対応地域は都道府県推定値にフォールバックします。キー未設定時は内蔵推定値を使用します。",
        "mlit_cities_cached": "市区町村レベルのデータを{n}地域分取得済みです。",
        "form_city_label": "市区町村",
        "bench_current": "{pref}: ¥{rate:,.0f}/㎡ · {src}",
        # Urban planning (都市計画) inputs
        "form_zoning": "用途地域",
        "form_address": "住所（位置情報用）",
        "form_address_help": "物件の完全な住所（例: 東京都文京区本駒込4-40-6）。物件の位置特定に使用します。任意ですが、地図・ハザード機能が有効になります。",
        "address_header": "📍 住所・位置",
        "address_tip": "💡 ローカル（VS Code）実行ではページ取得から自動入力されます。オンライン/モバイルでは、住所を貼り付けまたは入力して 📍 位置特定 をタップしてください。",
        "locate_btn": "📍 位置特定",
        "geocode_ok": "🗺️ 位置を特定しました: {lat:.5f}, {lon:.5f}。",
        "geocode_mock": "🗺️ デモ用座標 {lat:.5f}, {lon:.5f} — 実際の位置特定には GEOCODER='gsi' を設定してください。",
        "geocode_miss": "🗺️ この住所の位置を特定できませんでした — この物件では地図・ハザード機能を利用できません。",
        "hazard_flood": "洪水浸水想定区域",
        "hazard_landslide": "土砂災害警戒区域",
        "hazard_surge": "高潮浸水想定区域",
        "net_yield_delta": "諸費用控除後 実質{net}%",
        "net_yield_unknown": "実質利回り: 価格と表面利回りが必要です",
        "net_yield_breakdown": "実質利回りの内訳（{basis}）: 年間賃料{rent} − 空室{vac}% − 管理委託{mgmt}% − 管理費等{fees}/年（{src}） − 税{tax}/年 = NOI {noi}/年 → **実質{net}%**。前提はKPIウェイトタブで調整できます。",
        "invested_yield_note": "購入諸費用 {costs}（{pct}% — 仲介手数料3%+6万+税、不動産取得税、登録免許税、司法書士、印紙、保険）→ 総投資額 {invested} → **投下資本ベース{net_inv}%**。ポータルの表示は価格ベースのみです。",
        "area_sub_loan": "🚪 出口流動性リスク: {area}㎡は多くの投資用ローンが求める約{floor}㎡の下限を下回るため、売却先が現金購入者に限られ、売買スプレッドが広がります。スコアを{penalty}点減点しました。（単身世帯は増加傾向で賃貸需要自体は堅調です — これは融資と売却の制約です。）",
        "area_caution": "🚪 {area}㎡は約{caution}㎡未満で、融資を断る金融機関があり、東京23区のワンルーム条例も新築の最低面積をこの水準に定めています。売却先が限られるため、スコアを{penalty}点減点しました。",
        "area_investor": "🚪 一般的な投資用サイズです。{owner}㎡未満では実需層が住宅ローン控除を使えないため、売却先は投資家中心となります（この資産クラスでは標準的です）。",
        "area_broad": "🚪 {owner}㎡以上: 実需層が住宅ローン控除を利用できるため、投資家に加えてファミリー層も売却先となり、出口が最も広くなります。",
        "atier_sub_loan": "融資下限未満", "atier_caution": "融資注意水準",
        "atier_investor": "投資用標準", "atier_broad": "出口が広い", "atier_unknown": "不明",
        "exp_acq_label": "その他購入諸費用（価格の%・仲介手数料を除く）",
        "small_unit_pen_label": "20㎡未満の出口流動性減点（点）",
        "resale_horizon_label": "将来推計人口を読む期間（年）",
        "resale_severe": "📉 出口（売却）見通し: 周辺エリアの人口は{frm}年から{to}年で{pct}%と予測されています（公式500mメッシュ推計）。将来の購入者も入居者も減るため、出口の最大リスクです。",
        "resale_mild": "📉 出口（売却）見通し: 周辺人口は{frm}年から{to}年で{pct}%の予測です（公式500mメッシュ推計）。売却面でやや逆風です。",
        "resale_flat": "➡️ 出口（売却）見通し: 周辺人口は{frm}年から{to}年で{pct}%と概ね横ばいの予測です（公式500mメッシュ推計）。",
        "resale_growth": "📈 出口（売却）見通し: 周辺人口は{frm}年から{to}年で{pct}%の予測です（公式500mメッシュ推計）。人口が増えるエリアは賃料・売却の双方を支えます。",
        "resale_penalty_note": "出口流動性の減点を{extra}点引き上げました。",
        "resale_credit_note": "出口流動性の減点を{back}点引き下げました。",
        "resale_low_base": "📊 出口（売却）見通し: この約1km圏の推計人口は約{pop}人で、割合が意味を持つ最低水準（{minimum}人）を下回ります。河川・公園・鉄道用地・商業床が大半を占めるメッシュで典型的です。人口による調整は加減いずれも行いません。",
        "resale_anomaly": "📊 出口（売却）見通し: {frm}年→{to}年で{pct}%という推計が返りました（フィールド{field}）。人が住むエリアとして物理的にありえない値のため、データ異常として扱い、スコアには反映していません。念のため人口・ハザードマップで手動確認をおすすめします。",
        "resale_unknown": "📊 この地点の将来推計人口を取得できませんでした（MLITキーとデータ範囲が必要）。出口流動性は専有面積のみで評価し、人口による調整は加減いずれも行っていません。",
        "fees_actual": "入力値",
        "fees_estimated": "面積からの推定",
        "seismic_old": "🏚️ 旧耐震基準（{year}年頃築）: 金融機関が融資を断る・期間を短縮する場合が多く、売却時の流動性も低く、地震保険料も高くなります。耐震補強が必要な可能性もあります。スコアを{penalty}点減点しました。検査済証の確認と、金融機関への事前相談を強くおすすめします。",
        "seismic_grey": "⚠️ {year}年頃築 — ほぼ新耐震と考えられますが、1982〜83年竣工の建物は1981年6月1日より前の建築確認である場合があります。売主に建築確認日をご確認ください。",
        "hazard_tsunami": "津波浸水想定",
        "hzsev_rank1": "浸水深0.5m未満",
        "hzsev_rank2": "浸水深0.5～3m",
        "hzsev_rank3": "浸水深3～5m",
        "hzsev_rank4": "浸水深5～10m",
        "hzsev_rank5": "浸水深10～20m",
        "hzsev_rank6": "浸水深20m超",
        "hzsev_class1": "警戒区域",
        "hzsev_class2": "特別警戒区域",
        "hazard_warning": "🌊 ハザード: この所在地は{zones}に含まれます。収益物件として重大なリスクで、融資・保険料・売却に影響します。スコアを{penalty}点減点しました。購入前に必ず公式ハザードマップで確認してください。",
        "dev_kodo": "高度利用地区",
        "dev_road": "都市計画道路（約30m以内）",
        "dev_kodo_near": "高度利用地区（約300m以内・近接）",
        "dev_activity": "周辺約1kmで開発指定が平均より顕著に多い",
        "dev_chiku": "地区計画区域",
        "dev_chiku_near": "地区計画区域（約300m以内・近接）",
        "dev_context": "🏗️ 開発コンテキスト: {items}。再開発活動の先行指標です — スコアに+{bonus}点を加点しました（KPIウェイトタブで調整可能）。詳細は各区の都市計画課でご確認ください。",
        "form_zoning_help": "健美家・楽待の物件概要に記載されています。記載がなければ「不明」のまま — 不明は中立評価で、減点はされません。",
        "form_far": "指定容積率（%）",
        "form_far_help": "物件概要の容積率（例: 200）。記載がなければ0のままにしてください。",
        "zoning_warning": "⚠️ この用途地域は収益物件として重大なリスクがあります: 再建築や金融機関の融資が大きく制限されます。こうした地域の安価な高利回り物件は典型的な罠です — 購入前に必ず自治体に確認してください。",
        # Screenshot ingestion (vision extraction)
        "ext_header": "🧩 ブラウザ拡張から貼り付け",
        "ext_help": "ログイン状態でローカルのブラウザ拡張から物件を取得しましたか？そのJSONをここに貼り付けてください。ローカルで動作し、他の方法と同様にスコアリング前に確認できます。",
        "ext_paste_label": "拡張機能のJSONデータ",
        "ext_ingest_btn": "📥 貼り付けたデータを取り込む",
        "ext_ingest_failed": "データを読み取れませんでした（{msg}）。拡張機能がコピーしたJSONであることを確認してください。",
        "upload_header": "📷 または物件スクリーンショットをアップロード",
        "upload_label": "スマホのスクリーンショットをアップロード（RENOSY・健美家・楽待など）",
        "extract_btn": "🔎 スクリーンショットから物件情報を抽出",
        "extract_spinner": "スクリーンショットを読み取り中…",
        "extract_success": "{n}項目を抽出しました — 下のフォームを確認してからスコアリングしてください。",
        "extract_failed": "この画像から抽出できませんでした: {msg}",
        "warn_pref_unsupported": "都道府県「{name}」は本スクリーナーの対象（東京・神奈川・埼玉・千葉・大阪・京都・兵庫・愛知・福岡）外です — 手動で選択してください。対象外地域ではベンチマーク由来のスコアは参考になりません。",
        "warn_address_mismatch": "物件の都道府県（{listing_pref}）と一致しない{addr_pref}の住所を無視しました — ページのフッター/会社住所と思われ、物件の住所ではありません。必要に応じて手動で入力してください。",
        "warn_layout_unknown": "間取り「{name}」を物件種別に変換できませんでした — 手動で選択してください。",
        "warn_yield_computed": "利回りの記載がなかったため、賃料 ¥{rent:,.0f}/月 ÷ 価格 から {gy:.2f}% を算出しました。",
        "warn_mock_provider": "ビジョンプロバイダーが「mock」です — 表示中はデモ用の固定値で、スクリーンショットの内容ではありません。実際に抽出するには VISION_PROVIDER（openai / gemini / anthropic）と対応するAPIキーを設定してください。",
        "vision_note": "抽出にはVision LLMを使用します（VISION_PROVIDER と APIキーをシークレット/環境変数で設定）。抽出値はフォームに自動入力されるだけです — スコアリング前に必ず内容を確認・修正できます。",
        # URL import
        "url_import_header": "🔗 物件URLを貼り付け",
        "url_autofill_info": "URLそのものから自動入力: {items}。残りの項目はページ取得・スクリーンショット・手入力で補完してください。",
        "url_fetch_btn": "🌐 ページを取得して抽出",
        "url_fetch_spinner": "物件ページを取得中…",
        "url_fetch_note": "貼り付けたページ1件のみをユーザー操作で取得します — 一括スクレイピングではありませんが、ポータルの利用規約の確認はご自身の責任でお願いします。タイトル解析（価格・利回り）はAPIキー不要、全項目抽出は設定済みのLLMプロバイダーを使用します。ログインが必要なページ（楽待に多い）や取得がブロックされる場合は、スクリーンショットをご利用ください。",
        "url_fetch_failed": "ページを読み取れませんでした（{msg}）。スクリーンショットのアップロードをお試しください — そちらは確実に動作します。",
        "url_fetch_blocked": "このポータルは自動取得をブロックしました（楽待では一般的で、アプリの不具合ではなくポータルのアクセスポリシーです）。下の📷スクリーンショットをご利用ください — ご自身のブラウザから取得するためブロックされません。",
        "url_fetch_blocked_partial": "このポータルは自動取得をブロックしました（楽待では一般的）。URLから{items}を入力しました — 価格・面積・利回り等は下の📷スクリーンショットで補完してください。",
        "ward_resolved": "📍 **{ward}** のベンチマークでスコアリングしています（市区町村レベル）。",
        "ward_fallback": "📍 この住所の市区町村ベンチマークがないため、**{pref}** の都道府県平均でスコアリングしています。（ヒント: 住所に区名（例: 渋谷区）まで含めると、より局所的なスコアリングが有効になります。）",
        "ward_trend": "📈 {ward} の価格トレンド: 年率{pct:+.1f}%（{q}四半期・MLITサンプル{n}件に基づく）。表示のみ — スコアには未反映です。",
        "momentum_header": "📈 市区町村モメンタムランキング",
        "momentum_caption": "MLIT四半期中央値に基づく年率換算¥/m²トレンド — データが十分な{n}市区町村。参考情報です。ベンチマークを更新すると対象が広がります。",
        "momentum_axis": "年率換算価格トレンド（%/年）",
        # Market view
        "market_header": "📊 市場スキャン結果一覧",
        # Columns
        "col_prefecture": "都道府県",
        "col_rec_date": "推奨日",
        "col_orig_price": "推奨時価格(円)",
        "col_current_status": "現在のステータス",
        "col_days_listed": "掲載日数",
        "col_outcome": "結果",
        "col_factor": "ファクター",
        "col_weight": "ウェイト",
        "col_change": "初期値との差",
        # Statuses / outcomes
        "status_active": "掲載中",
        "status_sold": "成約 / 掲載終了",
        "status_price_drop": "値下げ",
        "status_unchanged": "掲載継続中",
        "outcome_win": "✅ 早期成約 — エンジンの判断が正解",
        "outcome_loss": "❌ 長期掲載/値下げ — エンジンにペナルティ",
        "outcome_neutral": "➖ シグナルなし",
        "outcome_pending": "⏳ リアリティチェック待ち",
        # Audit tab
        "audit_header": "🧠 ポートフォリオ監査 — ウォークフォワード追跡",
        "audit_intro": "追跡中の物件はURL・価格・日付とともにここに保存されます。ステータス確認では各物件ページを再訪し、掲載終了が確認できた場合は売却として扱います（早期は良好な判断、時間経過後は中立）。アクセス不可・ブロックされたページは掲載終了とはみなしません。手動で記録した結果が学習の基準データです。",
        "audit_empty": "まだ追跡中の物件がありません。物件を分析し「この物件を追跡」にチェックすると台帳に追加されます。",
        "audit_table_header": "推奨物件履歴台帳",
        "learning_header": "🤖 現在のシステムウェイト（エンジンの学習状況）",
        "learning_caption": "早期成約（{fast}日未満）はその物件の主要ファクターを上方修正し、値下げや{stale}日以上の掲載は下方修正します。ウェイトは下限{floor:.0%}で合計1に正規化されます。",
        "weight_history_header": "ウェイトの推移",
        "accuracy_label": "検証済み推奨",
        "checked_label": "評価済み件数",
        "winrate_label": "早期成約率",
        # Weights tab
        "weights_header": "⚙️ KPIウェイト・カスタマイザー",
        "gates_header": "🚦 立地ゲート（ハザード・開発）",
        "gates_caption": "これらはウェイト配分の外側で最終スコアに作用します: ハザードゾーンは最も深刻な区域の重大度に応じて減点（洪水は浸水深ランクで0.5～1.5倍、土砂災害は警戒0.8倍・特別警戒1.3倍、高潮・津波は1.0倍）、開発シグナルは加点。強いシグナル（高度利用地区・地区計画の区域内、または敷地に接する都市計画道路）は1フラグ、周辺シグナル（約300m以内の区域・周辺の開発密度が顕著）は0.5フラグとして計算し、最大2フラグ相当。正直な限界: MLITのAPIは都市再生特別地区・特定街区の指定を提供していません。これらは渋谷スクランブルスクエアや大阪ブリリアタワー堂島など日本最大級の再開発の多くを規定する制度であり、実際に再開発が進んでいてもここでは無反応となる場合があります。大規模タワーは各区の都市計画課で確認してください。「開発密度」の閾値は、分析済みタイルの中央値の2倍に自動調整されます（10タイル以上蓄積後）。",
        "hazard_pen_label": "ハザード減点（いずれかのゾーンに該当時、1回適用）",
        "seismic_pen_label": "旧耐震減点（1981年以前の建物）",
        "expenses_header": "💰 実質利回りの費用前提",
        "expenses_caption": "広告上の表面利回り（満室時・費用控除前）を、実際の投資判断に使えるNOIベースの実質利回りに変換するための前提です。物件の実際の管理費＋修繕積立金を入力すると、㎡単価による推定より優先されます。",
        "exp_fee_label": "管理費+修繕積立金の推定（円/㎡/月）",
        "exp_owner_label": "一棟・戸建の修繕負担（賃料の%）",
        "exp_vacancy_label": "想定空室率（%）",
        "exp_mgmt_label": "賃貸管理手数料（賃料の%）",
        "exp_tax_label": "固定資産税+都市計画税（年額・価格の%）",
        "dev_bonus_label": "開発加点（フラグごと、最大2フラグ）",
        "weights_intro": "このベースラインウェイトが100点満点の物件スコアを決定します。手動変更はウォークフォワードループと同じ `system_weights` 台帳に保存され、エンジンはあなたの設定値を起点に学習を続けます。",
        "weight_slider_help": "相対的な重要度 — 保存時に合計が1.0になるよう正規化されます。",
        "save_weights_btn": "💾 ウェイトを保存",
        "reset_weights_btn": "↩️ 初期値にリセット",
        "weights_saved": "ウェイトを保存し正規化しました。",
        "weights_reset": "ウェイトを初期値にリセットしました。",
        "current_weights_header": "適用中のウェイト",
        "note_manual": "手動調整",
        "note_reset": "初期値にリセット",
        "note_initial": "初期デフォルト",
        "note_walkforward": "リアリティチェック{n}件によるウォークフォワード更新",
        "formula_header": "スコアリング式",
        "formula_body": "スコア = 100 × (w_利回り × 利回りスコア + w_駅近 × 駅近スコア + w_築年 × 築年スコア + w_㎡単価 × ㎡単価スコア + w_開発 × 開発ポテンシャル)。利回りは4〜12%帯、駅徒歩は1〜20分帯（近いほど高評価）、築年数は0〜45年帯（新しいほど高評価、45年超は旧耐震ペナルティ）、㎡単価は都道府県基準単価に対する0.65〜1.35倍帯（相場より安いほど高評価）、開発ポテンシャルは用途地域（商業系=高評価、市街化調整区域・工業専用=ほぼゼロ）と指定容積率（100〜500%帯）の組み合わせで評価します。用途地域が不明の場合は中立の50点です。",
    },
}

def get_lang() -> str:
    return st.session_state.get("lang", "en")

def tr(key: str, **kwargs) -> str:
    """Look up a UI string for the active language, falling back to English."""
    table = TRANSLATIONS.get(get_lang(), TRANSLATIONS["en"])
    text = table.get(key) or TRANSLATIONS["en"].get(key, key)
    return text.format(**kwargs) if kwargs else text

def pref_name(key: str) -> str:
    """Translate an internal prefecture key (e.g. 'Tokyo') for display."""
    return PREF_NAMES.get(get_lang(), PREF_NAMES["en"]).get(key, key)

def factor_name(key: str) -> str:
    """Translate an internal factor key (e.g. 'yield_score') for display."""
    return FACTOR_NAMES.get(get_lang(), FACTOR_NAMES["en"]).get(key, key)

def ptype_name(key: str) -> str:
    """Translate an internal property-type key (e.g. '1R Mansion') for display."""
    return PTYPE_NAMES.get(get_lang(), PTYPE_NAMES["en"]).get(key, key)

def zoning_name(key: str) -> str:
    """Translate an internal zoning key (e.g. 'shogyo') for display."""
    return ZONING_NAMES.get(get_lang(), ZONING_NAMES["en"]).get(key, key)

def display_title(ptype: str, pref: str, station: float, age: float, area: float) -> str:
    """Build a fully localized listing title AT RENDER TIME.

    Titles are deliberately not baked into the DataFrame / DB rows in any one
    language: constructing them here means switching the sidebar language
    instantly re-renders every card, table and ledger row — no stale cached
    English fragments inside a Japanese UI (and vice versa)."""
    # Station and age can legitimately be UNKNOWN (NaN) now that the form offers
    # explicit unknown controls; int(NaN) raises. Render a dash rather than
    # inventing a number or crashing the card.
    def _n(v):
        f = safe_float(v)
        return "—" if math.isnan(f) else str(int(f))
    return tr("title_fmt", ptype=ptype_name(ptype), pref=pref_name(pref),
              station=_n(station), age=_n(age), area=_n(round(safe_float(area), 0)))

st.set_page_config(
    page_title="Japan Real Estate Quant Screener",
    page_icon="🏯",
    layout="wide",
    # Collapsed by default: on a phone an expanded sidebar covers the whole
    # screen. Users tap the ☰ menu to open the console.
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Styling & UI Engine
# ----------------------------------------------------------------------------
def inject_css(accent: str = "#e11d48", card_bg: str = "rgba(255,255,255,0.03)") -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --accent: {accent};
            --card-bg: {card_bg};
            --pos: #16a34a;
            --neg: #dc2626;
            --caution: #d97706;
            --plus: #4f46e5;
            --muted: #9ca3af;
            --muted: #94a3b8;
        }}
        /* Clear Streamlit's fixed top toolbar so the tab bar below the header
           doesn't slide underneath it. */
        .block-container,
        [data-testid="stMainBlockContainer"] {{ padding-top: 3.75rem; padding-bottom: 3rem; }}
        .qc-card {{
            background: var(--card-bg);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.10);
            transition: transform .12s ease, border-color .12s ease;
        }}
        .qc-card:hover {{ transform: translateY(-2px); border-color: var(--accent); }}
        .qc-label {{ font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
        .qc-value {{ font-size: 1.55rem; font-weight: 700; line-height: 1.15; margin-top: 2px; }}
        .qc-delta {{ font-size: 0.9rem; font-weight: 600; margin-top: 2px; }}
        .qc-pos {{ color: var(--pos); }}
        .qc-neg {{ color: var(--neg); }}
        .qc-pill {{ display:inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: 0.8rem; }}
        .qc-buy  {{ background: rgba(22,163,74,0.15);  color: var(--pos); }}
        .qc-hold {{ background: rgba(148,163,184,0.18); color: var(--muted); }}
        .qc-sell {{ background: rgba(220,38,38,0.15);  color: var(--neg); }}
        .qc-ticker {{ font-size: 1.15rem; font-weight: 800; }}
        .qc-sub {{ font-size: 0.82rem; color: var(--muted); }}
        @media (max-width: 640px) {{
            .qc-value {{ font-size: 1.3rem; }}
            .qc-ticker {{ font-size: 1.0rem; }}
            .qc-card {{ padding: 12px 14px; margin-bottom: 10px; }}
            .block-container {{ padding-left: .6rem; padding-right: .6rem; }}
            /* Streamlit does not stack st.columns on narrow screens by default,
               so multi-up card rows get badly squished on a phone.
               Force every column in a row to take the full width and wrap. */
            div[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: .4rem; }}
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }}
        }}
        /* --- Risk strip -----------------------------------------------------
           One compact chip per risk dimension, replacing what used to be up to
           seven stacked full-width banners. The left border carries the state:
           green clear, amber caution, red risk, indigo positive. "Unknown" is
           deliberately DASHED GREY and never green — a check we could not run
           must not look like a check that passed. Figures use tabular numerals
           so they align when the strip wraps.                                */
        .riskstrip {{ display: flex; flex-wrap: wrap; gap: .5rem;
                      margin: .35rem 0 .85rem; }}
        .rchip {{ display: flex; align-items: center; gap: .5rem;
                  padding: .45rem .7rem .45rem .6rem; border-radius: 8px;
                  background: var(--card-bg);
                  border: 1px solid rgba(128,128,128,.22);
                  border-left: 3px solid var(--muted);
                  font-size: .82rem; line-height: 1.15; }}
        .rchip__i {{ font-size: 1rem; }}
        .rchip__t {{ display: flex; flex-direction: column; }}
        .rchip__t b {{ font-weight: 600; letter-spacing: .01em; }}
        .rchip__t em {{ font-style: normal; opacity: .72; font-size: .74rem;
                        font-variant-numeric: tabular-nums; }}
        .rchip--ok {{ border-left-color: var(--pos); }}
        .rchip--caution {{ border-left-color: var(--caution); }}
        .rchip--risk {{ border-left-color: var(--neg);
                        background: rgba(220,38,38,.07); }}
        .rchip--plus {{ border-left-color: var(--plus); }}
        .rchip--unknown {{ border-left-style: dashed; opacity: .72; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _num_or_none(v):
    """Float for the DB, or NULL — never NaN, which SQLite stores as a value."""
    f = safe_float(v)
    return None if math.isnan(f) else float(f)

def listing_label(row) -> str:
    """A name a person can recognise.

    listing_id is a hash (MAN-4F2A9C71) — fine as a key, useless in a dropdown.
    Prefer the nickname you set, then the address, then the generated title, and
    only fall back to the id when there is genuinely nothing else."""
    get = row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row else d)
    for field in ("building_name", "nickname", "address", "title"):
        val = get(field)
        if val is not None and str(val).strip() and str(val).lower() != "nan":
            return str(val).strip()[:60]
    return str(get("listing_id") or "?")

def risk_strip(items: list[tuple[str, str, str, str]]) -> str:
    """Render the risk chips. Each item is (icon, label, value, state) where
    state is one of ok / caution / risk / plus / unknown.

    This replaces up to seven stacked banners with one scannable row. The point
    is that a clean listing should LOOK clean at a glance, while a real problem
    still reads as a problem — so state lives in colour and the detail moves one
    click away rather than disappearing."""
    allowed_states = {"ok", "caution", "risk", "plus", "unknown"}
    chips = "".join(
        f'<div class="rchip rchip--{state if state in allowed_states else "unknown"}">'
        f'<span class="rchip__i">{html.escape(str(icon))}</span>'
        f'<span class="rchip__t"><b>{html.escape(str(label))}</b><em>{html.escape(str(value))}</em></span></div>'
        for icon, label, value, state in items)
    return f'<div class="riskstrip">{chips}</div>'

def metric_card(label: str, value: str, delta: str | None = None, positive: bool | None = None) -> str:
    delta_html = ""
    if delta is not None:
        cls = "qc-pos" if positive else "qc-neg" if positive is not None else ""
        arrow = "▲ " if positive else "▼ " if positive is not None else ""
        delta_html = f'<div class="qc-delta {cls}">{arrow}{html.escape(str(delta))}</div>'
    return f'<div class="qc-card"><div class="qc-label">{html.escape(str(label))}</div><div class="qc-value">{html.escape(str(value))}</div>{delta_html}</div>'

def verdict_pill(score: float) -> str:
    """Map a 100-point composite onto a colored verdict pill (manual analyzer)."""
    if score >= get_verdict_strong():   cls, key = "qc-buy",  "verdict_strong"
    elif score >= get_verdict_consider(): cls, key = "qc-hold", "verdict_consider"
    else:                          cls, key = "qc-sell", "verdict_pass"
    return f'<span class="qc-pill {cls}">{tr(key)}</span>'

_SECRET_NAMES = ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "MLIT_API_KEY")

def _scrub_secrets(msg: str) -> str:
    """Redact any configured credential value from a user-facing message.
    Keys are sent in headers, which exception strings don't include, so this
    should never fire — it's defense in depth so the no-leak property keeps
    holding even if a future provider stub embeds a key somewhere visible."""
    for name in _SECRET_NAMES:
        val = _get_secret(name)
        if val and val in msg:
            msg = msg.replace(val, "***")
    return msg

def _get_secret(name: str, default: str | None = None) -> str | None:
    """Read a credential from Streamlit secrets first (set in the Cloud dashboard),
    then fall back to environment variables (handy for local runs). Never raises if
    no secrets file is configured."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)

# --- None-proofing Parsing Helpers ---
def far_or_nan(value) -> float:
    """Normalize a designated-FAR (容積率) value: a missing or non-positive FAR
    becomes NaN so analyze_properties._dev_score BYPASSES the FAR term (scoring
    on zoning class alone) instead of treating 0% as a real, floor-crushing
    value. Used everywhere FAR is persisted so live and re-scored picks agree."""
    v = safe_float(value, float("nan"))
    return v if (not math.isnan(v) and v > 0) else float("nan")

def safe_float(value, default: float = float("nan")) -> float:
    try:
        if value is None: return default
        if isinstance(value, str):
            # NFKC folds full-width digits (５８００ -> 5800); strip currency
            # marks, separators and any trailing unit so a hand-pasted
            # "5,800万円" or "１９.９８㎡" degrades to a number instead of NaN.
            value = unicodedata.normalize("NFKC", value)
            value = value.replace("¥", "").replace("円", "").replace(",", "").strip()
            for unit in ("万", "億", "%", "％", "m2", "m²", "㎡", "平米", "年", "分"):
                if value.endswith(unit):
                    value = value[: -len(unit)].strip()
            if value == "": return default
        f = float(value)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError): return default

def fmt_yen(v: float) -> str:
    if v is None or math.isnan(v): return "—"
    if abs(v) >= 1e8: return f"¥{v / 1e8:,.2f}億"
    return f"¥{v:,.0f}"

def fmt_pct(v: float) -> str: return "—" if math.isnan(v) else f"{v:.2f}%"
def fmt_num(v: float, nd: int = 1) -> str: return "—" if math.isnan(v) else f"{v:.{nd}f}"

# ----------------------------------------------------------------------------
# Database layer — system_weights + historical_picks
# ----------------------------------------------------------------------------
# Set when PRAGMA journal_mode=WAL is rejected (some network/cloud filesystems);
# surfaced as a sidebar warning so 'database is locked' issues are diagnosable.
_WAL_FALLBACK = False

# ---------------------------------------------------------------------------
# Backend selection. Default is local SQLite (zero-config, always works). Set
# DB_BACKEND="postgres" + SUPABASE_DB_URL (or PG* env) to persist to Supabase.
# The app NEVER hard-depends on Postgres: if the driver or URL is missing, it
# logs a one-time notice and falls back to SQLite, so a bad config degrades to
# "works locally" rather than "won't boot" — the same graceful-degradation
# philosophy used for MLIT, geocoding, and the vision providers.
# ---------------------------------------------------------------------------
def _db_backend() -> str:
    if (_get_secret("DB_BACKEND", "sqlite") or "sqlite").lower() == "postgres":
        try:
            import psycopg  # noqa: F401  (import test only)
            if _pg_dsn():
                return "postgres"
        except Exception:
            pass
        # requested but unusable -> note once, fall back
        if not st.session_state.get("_pg_fallback_noted"):
            st.session_state["_pg_fallback_noted"] = True
    return "sqlite"

def _pg_dsn() -> str | None:
    """Supabase connection string. Prefer a full URL; else assemble from parts.
    Supabase gives this under Project → Settings → Database → Connection string
    (use the 'Session'/pooler URI for Streamlit)."""
    url = _get_secret("SUPABASE_DB_URL") or _get_secret("DATABASE_URL")
    return url or None

# Translate the handful of SQLite-isms the app's SQL uses into Postgres form,
# so all ~30 call sites stay byte-for-byte identical across both backends.
def _sqlite_sql_to_pg(sql: str) -> str:
    s = sql.replace("?", "%s")
    s = s.replace("INSERT OR REPLACE INTO", "INSERT INTO")
    s = s.replace("IFNULL(", "COALESCE(")
    return s

class _PgCursorWrapper:
    """Makes a psycopg cursor behave like sqlite3's: .execute returns self so
    `.fetchone()/.fetchall()` chain, and rows are dict-like (name access)."""
    def __init__(self, cur): self._cur = cur
    def execute(self, sql, params=()):
        self._cur.execute(_sqlite_sql_to_pg(sql), params)
        return self
    def executemany(self, sql, seq):
        self._cur.executemany(_sqlite_sql_to_pg(sql), list(seq))
        return self
    def fetchone(self): return self._cur.fetchone()
    def fetchall(self): return self._cur.fetchall()
    def __iter__(self): return iter(self._cur.fetchall())

class _PgConnWrapper:
    """Adapts a psycopg connection to the sqlite3.Connection surface the app
    uses: `.execute(...).fetchone()`, `.executemany(...)`, dict-row access, and
    `.commit()`. ON CONFLICT is handled by the upsert helper below, not here."""
    def __init__(self, conn): self._conn = conn
    def execute(self, sql, params=()):
        cur = self._conn.cursor()
        return _PgCursorWrapper(cur).execute(sql, params)
    def executemany(self, sql, seq):
        cur = self._conn.cursor()
        return _PgCursorWrapper(cur).executemany(sql, seq)
    def commit(self): self._conn.commit()
    def close(self): self._conn.close()

# One long-lived Postgres connection per process, guarded by a lock. Opening a
# fresh TLS connection to Supabase per DB touch cost hundreds of ms, dozens of
# times per rerun — the "app became very slow" report. The lock serializes DB
# access across Streamlit's threads (psycopg connections aren't thread-safe);
# for a personal tool that's the right trade. Health-checked and re-opened on
# failure; rolled back on any exception so an aborted transaction can't poison
# subsequent queries.
_PG_CONN = None
_PG_LOCK = threading.Lock()

def _pg_connection():
    global _PG_CONN
    import psycopg
    from psycopg.rows import dict_row
    if _PG_CONN is None or _PG_CONN.closed:
        _PG_CONN = psycopg.connect(_pg_dsn(), row_factory=dict_row, autocommit=False)
    return _PG_CONN

@contextlib.contextmanager
def get_conn():
    """Yield a DB connection (SQLite or Postgres), commit on clean exit.

    SQLite: fresh connection per call (cheap locally), closed in finally.
    Postgres (Supabase): a REUSED singleton (see _pg_connection) — commit on
    clean exit, rollback + reconnect-next-time on error, never closed here.
    Every call site uses `with get_conn() as conn:` and is backend-agnostic.
    """
    global _WAL_FALLBACK, _PG_CONN
    if _db_backend() == "postgres":
        with _PG_LOCK:
            try:
                conn = _pg_connection()
            except Exception:
                _PG_CONN = None
                raise
            wrapped = _PgConnWrapper(conn)
            try:
                yield wrapped
                wrapped.commit()
            except Exception:
                try:
                    conn.rollback()          # clear aborted-transaction state
                except Exception:
                    _PG_CONN = None          # connection is dead; reconnect next call
                raise
        return
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # PRAGMA journal_mode does NOT raise when WAL is unavailable — SQLite
    # silently returns whatever mode it settled on. So detect by inspecting the
    # returned mode, not by catching an exception.
    try:
        mode = conn.execute("PRAGMA journal_mode=WAL;").fetchone()[0]
        conn.execute("PRAGMA synchronous=NORMAL;")
        if str(mode).lower() != "wal":
            _WAL_FALLBACK = True
    except sqlite3.Error:
        _WAL_FALLBACK = True
    try:
        yield conn
        conn.commit()          # preserve the transaction-commit the old `with conn` gave
    finally:
        conn.close()

def _upsert_sql(table: str, cols: list[str], conflict: list[str]) -> str:
    """Backend-correct upsert. SQLite keeps INSERT OR REPLACE; Postgres uses
    INSERT ... ON CONFLICT (...) DO UPDATE. Callers that previously wrote
    'INSERT OR REPLACE' now build their statement through this helper."""
    placeholders = ",".join("?" for _ in cols)
    collist = ",".join(cols)
    if _db_backend() == "postgres":
        updates = ",".join(f"{c}=EXCLUDED.{c}" for c in cols if c not in conflict)
        return (f"INSERT INTO {table} ({collist}) VALUES ({placeholders}) "
                f"ON CONFLICT ({','.join(conflict)}) DO UPDATE SET {updates}")
    return f"INSERT OR REPLACE INTO {table} ({collist}) VALUES ({placeholders})"

def _ensure_column(conn, table: str, column: str,
                   decl: str = "REAL DEFAULT 0.0") -> None:
    """Add `column` to `table` if absent (safe migration on both backends)."""
    if _db_backend() == "postgres":
        # Postgres declares types differently; map the SQLite decls the app uses.
        pg_decl = decl.replace("REAL", "DOUBLE PRECISION")
        conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {pg_decl}")
        return
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

def _ddl(sql: str) -> str:
    """Translate the app's SQLite DDL to Postgres when needed: SERIAL primary
    keys and DOUBLE PRECISION for REAL. Idempotent CREATE IF NOT EXISTS and the
    column set are identical across both."""
    if _db_backend() != "postgres":
        return sql
    return (sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
               .replace(" REAL", " DOUBLE PRECISION"))

# Bump when tables/columns change so existing databases re-run migrations once.
SCHEMA_VERSION = "19"

def init_db() -> None:
    """Create/migrate the schema. Fast path: every Streamlit page load is a NEW
    session, and the full DDL pass (~20 statements) cost real time per load on
    Supabase. A version marker in app_meta reduces the common case to ONE query;
    the full pass runs only on first boot or after a schema bump."""
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT value FROM app_meta WHERE key='schema_version'").fetchone()
        if row is not None and str(row["value"]) == SCHEMA_VERSION:
            return
    except Exception:
        pass   # app_meta missing (first boot / pre-versioning DB) -> full pass
    _init_db_full()
    with get_conn() as conn:
        conn.execute(_ddl("""
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )"""))
        conn.execute(_upsert_sql("app_meta", ["key", "value"], conflict=["key"]),
                     ("schema_version", SCHEMA_VERSION))
        conn.commit()

def _init_db_full() -> None:
    with get_conn() as conn:
        _x = lambda q: conn.execute(_ddl(q))
        # Long-format ledger: one row per factor per weight change, so the audit
        # tab can chart each factor's evolution independently.
        _x("""
            CREATE TABLE IF NOT EXISTS system_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                factor_name TEXT NOT NULL,
                current_weight REAL,
                note TEXT
            )""")
        # Every scan's Top-N picks land here with their exact URL / price / date.
        _x("""
            CREATE TABLE IF NOT EXISTS historical_picks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                listing_id TEXT NOT NULL,
                url TEXT NOT NULL,
                portal TEXT,
                title TEXT,
                ptype TEXT,
                prefecture TEXT,
                station_min REAL,
                building_age REAL,
                area_sqm REAL,
                price_yen REAL,
                gross_yield REAL,
                score REAL,
                factor_snapshot TEXT,        -- JSON of per-factor sub-scores (0-100)
                recommendation_date TEXT NOT NULL,
                current_status TEXT DEFAULT 'pending',
                current_price REAL,
                days_listed INTEGER,
                outcome INTEGER,             -- 1 win, -1 loss, 0 neutral, NULL pending
                evaluated INTEGER DEFAULT 0  -- each pick feeds the loop exactly once
            )""")
        # Safe migrations for pre-existing databases (no-ops on fresh installs).
        _ensure_column(conn, "system_weights", "note", "TEXT")
        _ensure_column(conn, "historical_picks", "evaluated", "INTEGER DEFAULT 0")
        # Property type stored alongside the title so the audit ledger can rebuild
        # a fully localized title at render time (older rows fall back to `title`).
        _ensure_column(conn, "historical_picks", "ptype", "TEXT")
        # Floor area for the ledger display and the price-efficiency factor.
        _ensure_column(conn, "historical_picks", "area_sqm", "REAL")
        # 都市計画 inputs behind the development-potential factor.
        _ensure_column(conn, "historical_picks", "zoning", "TEXT")
        _ensure_column(conn, "historical_picks", "far_pct", "REAL")
        # Geocoded coordinates (enabler for coordinate-keyed hazard/map layers).
        _ensure_column(conn, "historical_picks", "lat", "REAL")
        _ensure_column(conn, "historical_picks", "lon", "REAL")
        _ensure_column(conn, "historical_picks", "hazard_flags", "TEXT")
        _ensure_column(conn, "historical_picks", "dev_flags", "TEXT")
        _ensure_column(conn, "historical_picks", "net_yield", "REAL")
        _ensure_column(conn, "historical_picks", "monthly_fees_yen", "REAL")
        _ensure_column(conn, "historical_picks", "seismic", "TEXT")
        _ensure_column(conn, "historical_picks", "yield_basis", "TEXT")
        _ensure_column(conn, "historical_picks", "net_yield_invested", "REAL")
        _ensure_column(conn, "historical_picks", "area_tier", "TEXT")
        _ensure_column(conn, "historical_picks", "pop_outlook_pct", "REAL")
        _ensure_column(conn, "historical_picks", "nickname", "TEXT")
        # --- WS3: building identity (nullable; existing rows keep displaying) --
        for _c in ("building_name", "unit_number", "unit_floor", "source_document_name"):
            _ensure_column(conn, "historical_picks", _c, "TEXT")
        # --- WS5: how the data arrived, and who supplied it. These are separate
        #     questions: ingestion_channel is the route, source_type/agency_* is
        #     the party. Agency identity is provenance, never a score input.
        for _c in ("ingestion_channel", "source_type", "agency_name", "agent_name",
                   "agency_phone", "agency_address", "agency_role",
                   "source_document_date", "seller_name", "management_company",
                   "rent_guarantee_company", "extracted_entities", "provenance"):
            _ensure_column(conn, "historical_picks", _c, "TEXT")
        # --- WS: contractual rent is the base case; agent market rent is upside
        #     ONLY and must never overwrite it. Two columns, never one.
        _ensure_column(conn, "historical_picks", "monthly_rent_yen", "REAL")
        _ensure_column(conn, "historical_picks", "market_rent_yen", "REAL")
        _ensure_column(conn, "historical_picks", "occupancy", "TEXT")
        _ensure_column(conn, "historical_picks", "structure", "TEXT")
        _ensure_column(conn, "historical_picks", "size_resilience", "REAL")
        # v16: everything needed to REPRODUCE a size decision and a rent basis
        # from an audit row alone, without re-running the model.
        _ensure_column(conn, "historical_picks", "size_model_mode", "TEXT")
        _ensure_column(conn, "historical_picks", "size_adjustment", "REAL")
        _ensure_column(conn, "historical_picks", "size_evidence", "REAL")
        _ensure_column(conn, "historical_picks", "size_breakdown", "TEXT")
        _ensure_column(conn, "historical_picks", "rent_source", "TEXT")
        _ensure_column(conn, "historical_picks", "contractual_gross_pct", "REAL")
        _ensure_column(conn, "historical_picks", "gross_gap_pp", "REAL")
        # v17. source_url is its own column: a URL is not a document title, and a
        # SOURCE url (where evidence came from) is not the subject listing's
        # `url`. Three distinct things that were being collapsed into two.
        _ensure_column(conn, "historical_picks", "source_url", "TEXT")
        _ensure_column(conn, "historical_picks", "rent_evidence_summary", "TEXT")
        # v18: reviewed local market-rent evidence. NULL means unbenchmarked.
        _ensure_column(conn, "historical_picks", "local_median_rent_per_sqm", "REAL")
        _ensure_column(conn, "historical_picks", "rental_comparable_count", "INTEGER")
        _ensure_column(conn, "historical_picks", "rent_median_similarity", "REAL")
        _ensure_column(conn, "historical_picks", "rent_recency_months", "REAL")
        _ensure_column(conn, "historical_picks", "median_listing_duration_days", "REAL")
        _ensure_column(conn, "historical_picks", "rent_source_kind", "TEXT")
        _x("""
            CREATE TABLE IF NOT EXISTS rental_observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                historical_pick_id INTEGER,
                source_url TEXT,
                observed_at TEXT,
                monthly_rent_yen REAL NOT NULL,
                management_fee_yen REAL,
                total_monthly_cost_yen REAL,
                area_sqm REAL,
                floor_plan TEXT,
                unit_floor TEXT,
                building_age REAL,
                listing_status TEXT,
                evidence_kind TEXT,
                source_label TEXT,
                raw_context TEXT,
                selected INTEGER DEFAULT 0,
                confidence REAL,
                created_at TEXT
            )""")
        # v19: immutable post-screening revisions. historical_picks stays the
        # RECOMMENDATION snapshot — what was known when the call was made — and
        # every later edit appends here instead of overwriting it. Without this,
        # a recommendation becomes irreproducible the first time new evidence
        # arrives, and the walk-forward learner would be judging a call that was
        # never actually made.
        _x("""
            CREATE TABLE IF NOT EXISTS property_revisions (
                revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                historical_pick_id INTEGER NOT NULL,
                revision_number INTEGER NOT NULL,
                revision_kind TEXT NOT NULL,
                changed_at TEXT NOT NULL,
                changed_by TEXT,
                change_reason TEXT,
                source_name TEXT,
                source_url TEXT,
                source_document_name TEXT,
                ingestion_channel TEXT,
                fields_before TEXT,
                fields_after TEXT,
                changed_fields TEXT,
                provenance TEXT,
                refreshed_score REAL,
                refreshed_analysis TEXT,
                original_score_preserved INTEGER DEFAULT 1,
                UNIQUE (historical_pick_id, revision_number)
            )""")
        # manual=1 marks a real, user-entered listing (Analyze Property tab).
        # These are EXCLUDED from any automated status inference — assigning dice-
        # roll fates to real properties would corrupt both the ledger and the
        # learned weights. Live URL polling (roadmap step 6) will cover them.
        _ensure_column(conn, "historical_picks", "manual", "INTEGER DEFAULT 0")
        # The 5-digit 市区町村コード resolved for a manual/scan listing, so re-
        # scoring a ledger row reproduces the same hyper-local benchmark it was
        # originally scored against (NULL on legacy rows -> prefecture fallback).
        _ensure_column(conn, "historical_picks", "city_code", "TEXT")
        # LEGACY prefecture-level ¥/m² benchmarks. No longer written to (the
        # refresh now populates municipality_benchmarks below), but retained as a
        # graceful MIDDLE fallback tier for databases upgraded from the old
        # prefecture-only schema, and harmless (empty) on fresh installs.
        _x("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prefecture TEXT NOT NULL,
                sqm_rate REAL,
                sample_n INTEGER,
                source TEXT,
                updated_at TEXT
            )""")
        # Hyper-local ¥/m² benchmarks keyed by the 5-digit MLIT 市区町村コード.
        # One row per municipality (city_code is the PRIMARY KEY, so a refresh
        # upserts in place rather than stacking history). pref_code is the
        # leading 2-digit JIS code, kept denormalized for cheap per-prefecture
        # rollups / inspection.
        _x("""
            CREATE TABLE IF NOT EXISTS municipality_quarterly (
                city_code TEXT NOT NULL,
                year INTEGER NOT NULL,
                quarter INTEGER NOT NULL,
                median_ppsm REAL,
                sample_count INTEGER,
                updated_at TEXT,
                PRIMARY KEY (city_code, year, quarter)
            )""")
        _x("""
            CREATE TABLE IF NOT EXISTS dev_tile_stats (
                z INTEGER NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                kodo_count INTEGER,
                road_count INTEGER,
                checked_at TEXT,
                PRIMARY KEY (z, x, y)
            )""")
        _x("""
            CREATE TABLE IF NOT EXISTS municipality_benchmarks (
                city_code TEXT PRIMARY KEY,
                pref_code TEXT,
                avg_price_sqm REAL,
                sample_count INTEGER,
                updated_at TEXT
            )""")
        _x("""
            CREATE TABLE IF NOT EXISTS transaction_comparables (
                comp_id TEXT PRIMARY KEY, city_code TEXT NOT NULL, year INTEGER, quarter INTEGER,
                property_kind TEXT, floor_plan TEXT, area_sqm REAL, building_year INTEGER,
                structure TEXT, price_yen REAL, price_per_sqm REAL, fetched_at TEXT
            )""")
        for _c, _t in (("station_min", "REAL"), ("source_type", "TEXT"),
                       ("observed_at", "TEXT")):
            _ensure_column(conn, "transaction_comparables", _c, _t)
        # Indexes LAST: every table and _ensure_column'd column must exist first.
        # (Placed earlier, idx_picks_evaluated silently failed because `manual`
        # is added by _ensure_column further down, and idx_quarterly_city failed
        # because municipality_quarterly hadn't been created yet.) Failures are
        # tolerated but REPORTED, so a swallowed error can't hide again.
        for idx, tbl, cols in [
            ("idx_picks_evaluated", "historical_picks", "evaluated, manual"),
            ("idx_picks_listing", "historical_picks", "listing_id"),
            ("idx_picks_recdate", "historical_picks", "recommendation_date"),
            ("idx_picks_status", "historical_picks", "current_status"),
            ("idx_weights_ts", "system_weights", "factor_name, id"),
            ("idx_quarterly_city", "municipality_quarterly", "city_code"),
            ("idx_rentobs_pick", "rental_observations", "historical_pick_id"),
            ("idx_rentobs_url", "rental_observations", "source_url"),
            ("idx_revisions_pick", "property_revisions",
             "historical_pick_id, revision_number"),
            ("idx_revisions_changed_at", "property_revisions", "changed_at"),
            ("idx_comps_city", "transaction_comparables", "city_code, year, quarter"),
        ]:
            try:
                conn.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON {tbl}({cols})")
            except Exception as e:
                st.session_state.setdefault("_index_errors", []).append(f"{idx}: {e}")
        conn.commit()
    if get_weight_history().empty:
        save_weights(DEFAULT_WEIGHTS, note_key="note_initial")

def get_latest_weights() -> dict[str, float]:
    """Return the most recent weight per factor from system_weights, normalized.
    Session-memoized (invalidated by save_weights) — called several times per
    rerun and each call was a network round trip on Supabase."""
    return _memo("weights", _load_latest_weights)

def _load_latest_weights() -> dict[str, float]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT factor_name, current_weight FROM system_weights
            WHERE id IN (SELECT MAX(id) FROM system_weights GROUP BY factor_name)
        """).fetchall()
    raw = dict(DEFAULT_WEIGHTS)
    for r in rows:
        if r["factor_name"] in raw:
            w = safe_float(r["current_weight"], DEFAULT_WEIGHTS[r["factor_name"]])
            # Two integrity gates on anything read back from disk:
            #   w <= 0  -> migration artifact / corruption: substitute the default
            #   0 < w < MIN_WEIGHT -> manually edited below the floor: clamp up,
            # so the MIN_WEIGHT invariant holds even on a hand-edited database.
            if w <= 0:
                w = DEFAULT_WEIGHTS[r["factor_name"]]
            raw[r["factor_name"]] = max(MIN_WEIGHT, w)
    total = sum(raw.values()) or 1.0
    return {f: raw[f] / total for f in FACTORS}

def save_weights(weights: dict[str, float], note_key: str = "note_manual", **note_kwargs) -> None:
    """Persist one row per factor. The note stores the *key* (plus kwargs), not the
    rendered string, so the history table re-renders correctly when the user
    switches languages."""
    _memo_invalidate("weights")
    ts = datetime.now().isoformat(timespec="seconds")
    note = json.dumps({"key": note_key, "kwargs": note_kwargs})
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO system_weights (timestamp, factor_name, current_weight, note) VALUES (?,?,?,?)",
            [(ts, f, float(weights[f]), note) for f in FACTORS],
        )
        conn.commit()

def render_note(raw: str | None) -> str:
    """Render a stored note key into the active language (fallback: raw text)."""
    try:
        d = json.loads(raw or "{}")
        return tr(d.get("key", ""), **d.get("kwargs", {}))
    except Exception:
        return raw or ""

def _read_sql(sql: str, conn) -> pd.DataFrame:
    """pd.read_sql_query across both backends.

    Postgres subtlety: the connection uses dict_row (so the app's row["col"]
    access works everywhere), but pandas.read_sql_query CANNOT consume dict
    rows — it yields a frame with the right column names but blank/misaligned
    data (the "rows appear, columns empty" bug). So for Postgres we read through
    a fresh TUPLE cursor and build the DataFrame from cursor.description
    explicitly, which pandas-equivalent and correct."""
    if isinstance(conn, _PgConnWrapper):
        import psycopg
        from psycopg.rows import tuple_row
        with conn._conn.cursor(row_factory=tuple_row) as cur:
            cur.execute(_sqlite_sql_to_pg(sql))
            rows = cur.fetchall()
            cols = [d.name for d in cur.description] if cur.description else []
        return pd.DataFrame(rows, columns=cols)
    return pd.read_sql_query(sql, conn)

def get_weight_history() -> pd.DataFrame:
    with get_conn() as conn:
        df = _read_sql(
            "SELECT timestamp, factor_name, current_weight, note FROM system_weights ORDER BY id ASC",
            conn)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def get_historical_picks() -> pd.DataFrame:
    with get_conn() as conn:
        df = _read_sql("SELECT * FROM historical_picks ORDER BY id DESC", conn)
    return df

def save_manual_pick(r: dict) -> None:
    """Archive a user-entered REAL listing into historical_picks (manual=1).

    Shares the scan-pick schema — same factor snapshot, same ledger — but is
    flagged manual so automated status inference skips it; live URL polling
    will take ownership of these rows in roadmap step 6."""
    today = date.today().isoformat()
    snapshot = json.dumps({f: float(r[f]) for f in FACTORS})
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM historical_picks WHERE listing_id=? AND recommendation_date=?",
            (r["listing_id"], today),
        )
        conn.execute("""
            INSERT INTO historical_picks
                (listing_id, url, portal, title, ptype, prefecture, station_min, building_age,
                 area_sqm, price_yen, gross_yield, zoning, far_pct, city_code, lat, lon,
                 hazard_flags, dev_flags, net_yield, monthly_fees_yen, seismic,
                 yield_basis, net_yield_invested, area_tier, pop_outlook_pct, nickname,
                 building_name, unit_number, unit_floor, source_document_name,
                 ingestion_channel, source_type, agency_name, agent_name,
                 agency_phone, agency_address,
                 agency_role, source_document_date, seller_name,
                 management_company, rent_guarantee_company, extracted_entities,
                 provenance, monthly_rent_yen, market_rent_yen, occupancy,
                 structure, size_resilience,
                 size_model_mode, size_adjustment, size_evidence, size_breakdown,
                 rent_source, contractual_gross_pct, gross_gap_pp,
                 score, factor_snapshot,
                 recommendation_date, current_status, evaluated, manual)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'pending', 0, 1)
        """, (r["listing_id"], r["url"], r["portal"], r["title"], r["ptype"], r["prefecture"],
              float(r["station_min"]), float(r["building_age"]), float(r["area_sqm"]),
              float(r["price_yen"]), float(r["gross_yield"]), r.get("zoning") or "unknown",
              far_or_nan(r.get("far_pct")), r.get("city_code") or None,
              r.get("lat") if r.get("lat") is not None else None,
              r.get("lon") if r.get("lon") is not None else None,
              r.get("hazard_flags") or None,
              r.get("dev_flags") or None,
              None if math.isnan(safe_float(r.get("net_yield"))) else float(r["net_yield"]),
              None if math.isnan(safe_float(r.get("monthly_fees_yen"))) else float(r["monthly_fees_yen"]),
              r.get("seismic") or None,
              r.get("yield_basis") or None,
              None if math.isnan(safe_float(r.get("net_yield_invested"))) else float(r["net_yield_invested"]),
              r.get("area_tier") or None,
              None if math.isnan(safe_float(r.get("pop_outlook_pct"))) else float(r["pop_outlook_pct"]),
              r.get("nickname") or None,
              # WS3 identity + WS5 attribution + rent separation. All optional:
              # a listing typed in by hand simply leaves them null.
              *(r.get(_f) or None for _f in (
                  "building_name", "unit_number", "unit_floor", "source_document_name",
                  "ingestion_channel", "source_type", "agency_name", "agent_name",
                  "agency_phone", "agency_address",
                  "agency_role", "source_document_date", "seller_name",
                  "management_company", "rent_guarantee_company")),
              json.dumps(r["extracted_entities"], ensure_ascii=False)
                  if r.get("extracted_entities") else None,
              json.dumps(r["provenance"], ensure_ascii=False)
                  if r.get("provenance") else None,
              _num_or_none(r.get("monthly_rent_yen")),
              _num_or_none(r.get("market_rent_yen")),
              r.get("occupancy") or None,
              r.get("structure") or None,
              _num_or_none(r.get("size_resilience")),
              r.get("size_model_mode") or None,
              _num_or_none(r.get("size_adjustment")),
              _num_or_none(r.get("size_evidence")),
              r.get("size_breakdown") or None,
              r.get("rent_source") or None,
              _num_or_none(r.get("contractual_gross_pct")),
              _num_or_none(r.get("gross_gap_pp")),
              float(r["score"]), snapshot, today))
        conn.commit()

# ----------------------------------------------------------------------------
# Real ¥/m² benchmarks — MLIT 不動産情報ライブラリ (Real Estate Info Library)
# ----------------------------------------------------------------------------
def save_benchmark(prefecture: str, sqm_rate: float, sample_n: int, source: str) -> None:
    """LEGACY prefecture-level writer. Kept for the upgrade/fallback path; the
    live refresh now writes municipality rows via save_municipality_benchmark."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO benchmarks (prefecture, sqm_rate, sample_n, source, updated_at) VALUES (?,?,?,?,?)",
            (prefecture, float(sqm_rate), int(sample_n), source, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
    _memo_invalidate("pref_bench")

def save_municipality_benchmark(city_code: str, pref_code: str,
                                avg_price_sqm: float, sample_count: int) -> None:
    """Upsert one municipality's ¥/m² benchmark. city_code is the PRIMARY KEY,
    so a re-run REPLACES the row rather than stacking history — the table always
    holds exactly one current figure per 市区町村."""
    with get_conn() as conn:
        conn.execute(
            _upsert_sql("municipality_benchmarks",
                        ["city_code", "pref_code", "avg_price_sqm", "sample_count", "updated_at"],
                        conflict=["city_code"]),
            (str(city_code), str(pref_code), float(avg_price_sqm), int(sample_count),
             datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
    _memo_invalidate("bench_count", "muni_bench")

def _memo(key: str, loader):
    """Session-scoped memo for per-rerun DB reads. On Supabase every query is a
    network round trip, and these values change only on explicit writes — so
    the write sites call _memo_invalidate and everything else reads the cache."""
    cache = st.session_state.setdefault("_db_memo", {})
    if key not in cache:
        cache[key] = loader()
    return cache[key]

def _memo_invalidate(*keys: str) -> None:
    cache = st.session_state.setdefault("_db_memo", {})
    if not keys:
        cache.clear()
    for k in keys:
        cache.pop(k, None)

def count_municipality_benchmarks() -> int:
    """How many municipalities currently carry a real (>0) city-level benchmark
    — surfaced in the sidebar so users can see coverage at a glance.
    Session-memoized (invalidated on benchmark refresh)."""
    return _memo("bench_count", _load_bench_count)

def _load_bench_count() -> int:
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM municipality_benchmarks WHERE avg_price_sqm > 0"
            ).fetchone()
        return int(row["n"] if row else 0)
    except Exception:
        return 0

def save_quarterly_median(city_code: str, year: int, quarter: int,
                          median_ppsm: float, n: int) -> None:
    """Upsert one ward-quarter median. These rows are the raw material for the
    momentum signal; the pooled benchmark in municipality_benchmarks stays the
    scoring denominator."""
    with get_conn() as conn:
        conn.execute(
            _upsert_sql("municipality_quarterly",
                        ["city_code", "year", "quarter", "median_ppsm", "sample_count", "updated_at"],
                        conflict=["city_code", "year", "quarter"]),
            (city_code, int(year), int(quarter), float(median_ppsm), int(n),
             datetime.now().isoformat(timespec="seconds")))
    _memo_invalidate("momentum_table")

def get_ward_momentum(city_code: str | None) -> tuple[float, int, int] | None:
    """Annualized ward price trend from stored quarterly medians.

    Returns (annualized_pct, quarters_used, total_samples) or None when fewer
    than 4 sample-gated quarters exist. Method: order the quarters, compare the
    mean of the later half against the mean of the earlier half, and annualize
    by the midpoint gap. Deliberately simple and DISPLAY-ONLY for now — it does
    not enter the score until the signal has been eyeballed against reality.
    Quarterly rows below MLIT_MIN_QTR_SAMPLES were never stored, so thin-ward
    noise can't masquerade as momentum."""
    if not city_code:
        return None
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT year, quarter, median_ppsm, sample_count FROM municipality_quarterly "
            "WHERE city_code=? ORDER BY year, quarter", (city_code,)).fetchall()
    if len(rows) < 4:
        return None
    meds = [float(r["median_ppsm"]) for r in rows]
    ns = [int(r["sample_count"]) for r in rows]
    return _momentum_from_series(meds, ns)

def _momentum_from_series(meds: list[float], ns: list[int]) -> tuple[float, int, int] | None:
    """Shared momentum math: later-half mean vs earlier-half mean of the
    chronologically ordered quarterly medians, annualized by the midpoint gap.
    None below 4 quarters (the sample gate lives at WRITE time)."""
    if len(meds) < 4:
        return None
    half = len(meds) // 2
    first, last = meds[:half], meds[half:]
    a, b = sum(first) / len(first), sum(last) / len(last)
    if a <= 0:
        return None
    mid_first = (half - 1) / 2.0
    mid_last = half + (len(last) - 1) / 2.0
    gap_q = max(mid_last - mid_first, 1.0)
    annualized = (b / a - 1.0) * (4.0 / gap_q) * 100.0
    return round(annualized, 1), len(meds), sum(ns)

def get_all_ward_momentum() -> pd.DataFrame:
    return _memo("momentum_table", _load_all_ward_momentum)

def _load_all_ward_momentum() -> pd.DataFrame:
    """Every ward with a computable momentum, as a ranked DataFrame:
    columns [city_code, ward, pref, pct, quarters, samples], sorted desc by pct.
    Empty DataFrame when no ward has 4+ gated quarters yet.

    Performance note: fetches ALL quarterly rows in ONE query and groups in
    pandas. The previous version issued one query per ward (1+N round trips) —
    invisible on local SQLite, seconds of latency against Supabase."""
    with get_conn() as conn:
        df = _read_sql(
            "SELECT city_code, year, quarter, median_ppsm, sample_count "
            "FROM municipality_quarterly ORDER BY city_code, year, quarter", conn)
    rows = []
    if not df.empty:
        for cc, g in df.groupby("city_code", sort=False):
            mom = _momentum_from_series([float(v) for v in g["median_ppsm"]],
                                        [int(v) for v in g["sample_count"]])
            if mom:
                pct, q, n = mom
                rows.append({"city_code": cc, "ward": municipality_label(cc),
                             "pref": pref_name(_pref_key_from_city_code(cc) or ""),
                             "pct": pct, "quarters": q, "samples": n})
    if not rows:
        return pd.DataFrame(columns=["city_code", "ward", "pref", "pct", "quarters", "samples"])
    return pd.DataFrame(rows).sort_values("pct", ascending=False).reset_index(drop=True)

def get_sqm_rate(city_code: str | None, pref: str | None = None) -> tuple[float, str, int]:
    """Return (rate, source, sample_n) for the ¥/m² benchmark to score against,
    resolved at the FINEST granularity available. Three graceful tiers:

        1. MLIT_CITY  — a real municipality benchmark for `city_code`
                        (municipality_benchmarks). This is the upgrade's point:
                        Shibuya is compared to Shibuya, not a Tokyo-wide blend.
        2. MLIT_PREF  — a legacy prefecture-level MLIT row (only present in DBs
                        upgraded from the old schema), a coarse but real figure.
        3. default    — the built-in PREF_PROFILES estimate, so the
                        price-efficiency factor ALWAYS has a denominator and the
                        scoring engine never breaks on a rare/unmapped location.

    `pref` (internal key) is optional: when omitted it is recovered from
    city_code's leading two digits. Either argument may be None.

    Performance: both benchmark tables are tiny (dozens of rows) but were being
    queried PER CALL — the sidebar alone made 6 calls per rerun, and scans one
    per distinct ward. On Supabase every query is a network round trip, so both
    tables are now loaded ONCE per session (memoized, invalidated on refresh)
    and the tiers resolve in memory."""
    muni = _memo("muni_bench", _load_muni_bench)
    # Tier 1 — municipality-level real data.
    if city_code and str(city_code) in muni:
        rate, n = muni[str(city_code)]
        if rate > 0:
            return rate, "MLIT_CITY", n

    pref_key = pref or _pref_key_from_city_code(city_code)

    # Tier 2 — legacy prefecture-level real data (upgrade path only).
    legacy = _memo("pref_bench", _load_pref_bench)
    if pref_key and pref_key in legacy:
        rate, n = legacy[pref_key]
        if rate > 0:
            return rate, "MLIT_PREF", n

    # Tier 3 — built-in estimate constant.
    fallback = PREF_PROFILES.get(pref_key, {}).get("sqm_rate", 500_000)
    return float(fallback), "default", 0

def _load_muni_bench() -> dict[str, tuple[float, int]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT city_code, avg_price_sqm, sample_count FROM municipality_benchmarks"
        ).fetchall()
    return {str(r["city_code"]): (safe_float(r["avg_price_sqm"], 0.0),
                                  int(r["sample_count"] or 0)) for r in rows}

def _load_pref_bench() -> dict[str, tuple[float, int]]:
    """Latest legacy row per prefecture (id DESC wins)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT prefecture, sqm_rate, sample_n FROM benchmarks ORDER BY id ASC"
        ).fetchall()
    out: dict[str, tuple[float, int]] = {}
    for r in rows:   # ascending order -> later rows overwrite = latest wins
        out[str(r["prefecture"])] = (safe_float(r["sqm_rate"], 0.0), int(r["sample_n"] or 0))
    return out

def _benchmark_src_label(src: str, n: int) -> str:
    """Localized provenance label for a get_sqm_rate() source, centralized so
    the sidebar and analyzer render the city / prefecture / estimate tiers
    consistently (and so new tiers stay mirrored across EN/JA in one place)."""
    if src == "MLIT_CITY":
        return tr("benchmark_src_mlit", n=n)
    if src == "MLIT_PREF":
        return tr("benchmark_src_mlit_pref", n=n)
    return tr("benchmark_src_default")

def _parse_mlit_payload(payload: dict) -> list[float]:
    """Extract sane ¥/m² samples from an MLIT XIT001 response payload.

    Kept separate from the network call so it is unit-testable offline. Filters
    to building-inclusive residential transaction types and rejects ¥/m² values
    outside MLIT_PPSM_BOUNDS (data-entry outliers, farmland, etc.)."""
    ppsms: list[float] = []
    for d in payload.get("data", []) or []:
        price = safe_float(str(d.get("TradePrice", "")).replace(",", ""))
        area = safe_float(str(d.get("Area", "")).replace(",", ""))
        kind = str(d.get("Type", ""))
        if math.isnan(price) or math.isnan(area) or area <= 0:
            continue
        # 中古マンション等 (pre-owned condos) and 宅地(土地と建物) (land+building)
        # are the categories comparable to investable listings.
        if not any(k in kind for k in ("中古マンション", "宅地")):
            continue
        ppsm = price / area
        if MLIT_PPSM_BOUNDS[0] <= ppsm <= MLIT_PPSM_BOUNDS[1]:
            ppsms.append(ppsm)
    return ppsms

# Japanese era -> Gregorian offsets. `元年` means year 1 of an era.
_ERA_OFFSETS = {"令和": 2018, "平成": 1988, "昭和": 1925, "大正": 1911, "明治": 1867}

# ---------------------------------------------------------------------------
# WS4: PDF / image / multi-page agent-document ingestion
# ---------------------------------------------------------------------------
# Text PDFs are read directly; scanned ones are rendered to images and sent to
# the configured vision provider. Uploaded bytes are held in memory for the
# duration of the extraction only — nothing is persisted unless the user
# explicitly asks, which is the "no unintended persistence" requirement.
MAX_UPLOAD_MB = 20
MAX_PDF_PAGES = 8
PDF_RENDER_DPI = 150
# Below this many extracted characters a page is treated as scanned/graphical and
# rendered for the vision model. Real agent pages carry hundreds of characters;
# a near-empty result means the text lives in an image. Set deliberately low so a
# text page is never sent to a paid vision call unnecessarily.
MIN_TEXT_CHARS_PER_PAGE = 40

class DocumentError(Exception):
    """Unreadable/oversized/encrypted upload. Always surfaced, never silent."""

def pdf_available() -> tuple[bool, str]:
    try:
        import pymupdf  # noqa: F401
        return True, ""
    except Exception:
        try:
            import fitz  # noqa: F401
            return True, ""
        except Exception as e:
            return False, str(e)

def _open_pdf(data: bytes):
    try:
        import pymupdf as _pdf
    except Exception:
        import fitz as _pdf
    return _pdf.open(stream=data, filetype="pdf")

# ---------------------------------------------------------------------------
# Select All / Copy ingestion — parsing a listing page the USER copied
# ---------------------------------------------------------------------------
# Some portals block server-side fetches. Rather than work around that, this
# route asks the person to copy the page they are already viewing and paste it
# in. Nothing is requested from the portal, and the pasted text is held only for
# the length of the extraction unless they explicitly choose to keep it.
#
# The text is UNTRUSTED: it prefills the editable review form and never scores,
# saves or modifies a tracked record on its own.
ORIENTATION_MAP = {"東": "east", "西": "west", "南": "south", "北": "north",
                   "南東": "southeast", "南西": "southwest",
                   "北東": "northeast", "北西": "northwest"}
MANAGEMENT_TYPE_MAP = {"自主管理": "self_managed", "全部委託": "full_outsourced",
                       "一部委託": "partial_outsourced", "委託": "outsourced"}
LAND_RIGHT_MAP = {"所有権": "ownership", "借地権": "leasehold",
                  "定期借地権": "fixed_term_leasehold"}
AGENCY_ROLE_JP_MAP = {"仲介": "intermediary", "媒介": "intermediary",
                      "売主": "seller", "代理": "seller_agent"}
# Rent labels that assert a VERIFIED current lease. Everything else — 想定, 参考,
# 相場, 予想 — is an assumption, no matter what 現況 says elsewhere on the page.
VERIFIED_RENT_LABELS = ("契約賃料", "現行賃料", "現況賃料", "現行家賃", "契約家賃")

def _pt_num(text, pattern, cast=float):
    m = re.search(pattern, text)
    if not m:
        return None
    raw = next((g for g in m.groups() if g), None)
    if raw is None:
        return None
    try:
        return cast(str(raw).replace(",", "").strip())
    except Exception:
        return None

def parse_pasted_listing(pasted: str) -> dict:
    """Deterministic parse of copied portal text (Rakumachi and similar).

    Label-anchored throughout: the standard Japanese portal layout is regular
    enough that no model is needed, and a deterministic answer is auditable in a
    way a model's is not. A text LLM may fill what remains, but never overrides
    a label-anchored value.
    """
    raw_text = pasted or ""
    text = unicodedata.normalize("NFKC", raw_text)
    out: dict = {}

    def put(key, value):
        if value is not None and value != "" and out.get(key) in (None, "", [], {}):
            out[key] = value

    # --- category, title, building name -------------------------------------
    if re.search(r"区分マンション", text):
        put("property_category", "区分マンション")
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    for ln in lines[:4]:
        flat_ln = unicodedata.normalize("NFKC", ln)
        # A marketing headline is a SENTENCE about the deal, not a building
        # name: it advertises distance, tenancy or financing and usually ends in
        # ！. Treating it as building_name mislabels the property permanently.
        if re.search(r"[！!]|徒歩\s*\d+\s*分|オーナーチェンジ|融資|相談可", flat_ln) \
                and not re.match(r"^(?:区分マンション|一棟|土地|戸建)", flat_ln):
            put("listing_title", ln)
            break
    bn = re.search(r"(?:物件名|建物名|マンション名)\s*[:：]?\s*([^\n]{2,60})", raw_text)
    if bn:
        put("building_name", bn.group(1).strip())
    # building_name otherwise stays None on purpose — see the test fixture.

    # --- money ---------------------------------------------------------------
    put("price_man", _pt_num(text, r"(?:販売価格|物件価格|価格)[^\d]{0,10}([\d,]+)\s*万円"))
    put("gross_yield_pct", _pt_num(text, r"(?:表面利回り|想定利回り|満室時利回り)[^\d]{0,10}([\d.]+)\s*%"))
    annual = _pt_num(text, r"想定年間収入[^\d]{0,10}([\d,]+)\s*円", int)
    monthly = _pt_num(text, r"\(\s*([\d,]+)\s*円\s*/\s*月\s*\)", int)
    if monthly is None:
        monthly = _pt_num(text, r"(?:想定月額収入|想定賃料|想定家賃)[^\d]{0,10}([\d,]+)\s*円", int)
    if annual is not None:
        put("expected_annual_income_yen", annual)
    if monthly is not None:
        # An ASSUMPTION, so it goes to market rent. 現況 賃貸中 says a tenant
        # exists; it does not reveal what that tenant actually pays.
        put("market_rent_yen", monthly)
    elif annual is not None:
        put("market_rent_yen", int(round(annual / 12.0)))
    for lbl in VERIFIED_RENT_LABELS:
        v = _pt_num(text, lbl + r"[^\d]{0,10}([\d,]+)\s*円", int)
        if v:
            put("monthly_rent_yen", v)
            break
    mgmt = _pt_num(text, r"管理費(?:\s*\(月額\))?[^\d]{0,10}([\d,]+)\s*円", int)
    reserve = _pt_num(text, r"修繕積立金(?:\s*\(月額\))?[^\d]{0,10}([\d,]+)\s*円", int)
    if mgmt is not None:
        put("management_fee_yen", mgmt)
    if reserve is not None:
        put("repair_reserve_yen", reserve)
    if mgmt is not None or reserve is not None:
        # One-off settlement amounts (修繕準備金 etc.) are deliberately excluded.
        put("monthly_fees_yen", (mgmt or 0) + (reserve or 0))

    # --- location + stations --------------------------------------------------
    addr = re.search(r"所在地\s*[:：]?\s*([^\n]{4,60})", raw_text)
    if addr:
        a = addr.group(1).strip()
        put("address_jp", a)
        pm = re.match(r"((?:東京都|北海道|(?:京都|大阪)府|.{2,3}県))", a)
        if pm:
            put("prefecture_jp", pm.group(1))
            rest = a[len(pm.group(1)):]
            cm = re.match(r"(.{1,8}?市.{1,6}?区|.{1,8}?市|.{1,6}?区|.{1,8}?郡.{1,8}?町)", rest)
            if cm:
                put("city_jp", cm.group(1))
    pairs = re.findall(r"([^\s/／\n]{2,12}?駅)\s*徒歩\s*(\d{1,2})\s*分", text)
    if pairs:
        # The same station usually appears twice — once in the marketing
        # headline and once in the 交通 row. Deduplicate on (station, minutes)
        # so the option list reflects genuinely distinct stations.
        opts, seen_st = [], set()
        for st_, w in pairs:
            key = (st_, int(w))
            if key in seen_st:
                continue
            seen_st.add(key)
            opts.append({"station": st_, "walk_min": int(w)})
        put("station_options", opts)
        put("station_walk_min", min(o["walk_min"] for o in opts))

    # --- building attributes --------------------------------------------------
    stm = re.search(r"建物構造\s*[:：]?\s*(SRC造|RC造|鉄骨造|軽量鉄骨造|木造)", text)
    if stm:
        put("structure", {"SRC造": "SRC", "RC造": "RC", "鉄骨造": "S",
                          "軽量鉄骨造": "S", "木造": "Wood"}[stm.group(1)])
    built = re.search(r"築年月\s*[:：]?\s*(\d{4})\s*年\s*(\d{1,2})\s*月", text)
    if built:
        y, mo = int(built.group(1)), int(built.group(2))
        put("construction_date", f"{y:04d}-{mo:02d}")
        today = date.today()
        put("building_age_years", today.year - y - (today.month < mo))
        shown = _pt_num(text, r"築\s*(\d{1,2})\s*年", int)
        if shown is not None and abs(shown - out["building_age_years"]) > 1:
            # Keep the precise date and SAY they disagree rather than choosing.
            put("age_discrepancy", {"displayed": shown,
                                    "calculated": out["building_age_years"]})
    for label, key, mapping in (("土地権利", "land_right", LAND_RIGHT_MAP),
                                ("管理形態", "management_type", MANAGEMENT_TYPE_MAP),
                                ("取引態様", "agency_role", AGENCY_ROLE_JP_MAP)):
        m = re.search(label + r"\s*[:：]?\s*([^\n(（]{1,12})", text)
        if m:
            token = m.group(1).strip()
            for jp, val in mapping.items():
                if jp in token:
                    put(key, val)
                    break
    put("area_sqm", _pt_num(text, r"専有面積[^\d]{0,10}([\d.]+)\s*(?:m2|m²|㎡)"))
    put("balcony_sqm", _pt_num(text, r"バルコニー面積[^\d]{0,10}([\d.]+)\s*(?:m2|m²|㎡)"))
    lay = re.search(r"間取り\s*[:：]?\s*([1-9]?(?:R|K|DK|LDK|SLDK))", text)
    if lay:
        put("layout", lay.group(1))
    ori = re.search(r"方角\s*[:：]?\s*([東西南北]{1,2})", text)
    if ori:
        put("orientation", ORIENTATION_MAP.get(ori.group(1)))
    fl = re.search(r"階数\s*[:：]?\s*(\d{1,2})\s*階\s*[／/]\s*(\d{1,2})\s*階建", text)
    if fl:
        put("unit_floor", int(fl.group(1)))
        put("total_floors", int(fl.group(2)))
    put("total_units", _pt_num(text, r"総戸数[^\d]{0,10}([\d,]+)\s*戸", int))
    zm = re.search(r"用途地域\s*[:：]?\s*([^\n]{2,12})", text)
    if zm:
        put("zoning_jp", zm.group(1).strip())
    if re.search(r"現況\s*[:：]?\s*(?:賃貸中|入居中)", text):
        put("occupancy", "rented")
    elif re.search(r"現況\s*[:：]?\s*空室", text):
        put("occupancy", "vacant")
    hm = re.search(r"引渡(?:可能年月|し)\s*[:：]?\s*([^\n]{1,12})", text)
    if hm:
        put("handover", "consultation" if "相談" in hm.group(1) else hm.group(1).strip())
    mn = re.search(r"管理番号\s*[:：]?\s*([A-Za-z0-9\-]{3,24})", text)
    if mn:
        put("management_number", mn.group(1))
    for label, key in (("情報登録日", "registered_at"), ("更新日", "updated_at"),
                       ("次回更新予定日", "next_update_at")):
        m = re.search(label + r"\s*[:：]?\s*(\d{4})\s*[/年]\s*(\d{1,2})\s*[/月]\s*(\d{1,2})", text)
        if m:
            put(key, f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}")
    # Companies ONLY when their role is printed. Never inferred.
    for label in ("売主", "仲介会社", "管理会社", "保証会社"):
        m = re.search(label + r"\s*[:：]\s*([^\n]{2,40})", raw_text)
        if m:
            out.setdefault("company_roles", []).append([label, m.group(1).strip()])

    # Portal only from an explicit marker — never from generic property wording.
    for marker, portal in (("楽待", "Rakumachi"), ("rakumachi", "Rakumachi"),
                           ("健美家", "Kenbiya"), ("kenbiya", "Kenbiya"),
                           ("RENOSY", "Renosy"), (" renosy", "Renosy")):
        if marker.lower() in text.lower():
            put("portal_guess", portal)
            break
    if out.get("management_number", "").startswith("RR-"):
        put("portal_guess", "Rakumachi")

    out["ingestion_channel"] = "pasted_text"
    out["source_type"] = "property_portal"
    out["extraction_method"] = "pasted_text"
    return out

def pasted_listing_warnings(fields: dict) -> list[dict]:
    """Review warnings a human must see BEFORE scoring pasted text."""
    warns = []
    if fields.get("occupancy") == "rented" and not fields.get("monthly_rent_yen") \
            and fields.get("market_rent_yen"):
        warns.append({"key": "paste_warn_assumed_rent",
                      "kwargs": {"amount": f"{int(fields['market_rent_yen']):,}"}})
    ann, mon = fields.get("expected_annual_income_yen"), fields.get("market_rent_yen")
    if ann and mon:
        # Allow a month of rounding; anything larger is a real inconsistency.
        if abs(ann - mon * 12) > max(12, mon * 0.02):
            warns.append({"key": "paste_warn_income_mismatch",
                          "kwargs": {"annual": f"{int(ann):,}",
                                     "monthly": f"{int(mon):,}",
                                     "implied": f"{int(mon * 12):,}"}})
    if fields.get("age_discrepancy"):
        d = fields["age_discrepancy"]
        warns.append({"key": "paste_warn_age_conflict",
                      "kwargs": {"shown": d["displayed"], "calc": d["calculated"]}})
    for critical in ("price_man", "area_sqm"):
        if not fields.get(critical):
            warns.append({"key": "paste_warn_missing", "kwargs": {"field": critical}})
    return warns

def extract_listing_from_text(text: str) -> dict:
    """Extract listing fields from a TEXT pdf page without a vision call.

    Reuses the same field patterns as the rest of the app so a text PDF costs
    nothing and behaves identically. Also collects labelled company/role pairs
    for map_company_roles() — labels only; an unlabelled company is never
    promoted to an agency here."""
    n = unicodedata.normalize("NFKC", text or "")
    out: dict = {}
    def grab(pat, cast=float):
        mm = re.search(pat, n)
        if not mm:
            return None
        raw = next((g for g in mm.groups() if g), None)   # alternation-safe
        if raw is None:
            return None
        try:
            return cast(str(raw).replace(",", ""))
        except Exception:
            return None
    oku = re.search(r"([\d,.]+)\s*億\s*(?:([\d,]+)\s*万)?円", n)
    if oku:
        out["price_man"] = (float(oku.group(1).replace(",", "")) * 10000
                            + (float(oku.group(2).replace(",", "")) if oku.group(2) else 0))
    else:
        v = grab(r"(?:販売価格|物件価格|価格)[^\d]{0,8}([\d,]+)\s*万円")
        if v is not None:
            out["price_man"] = v
        else:
            y = grab(r"(?:販売価格|物件価格|価格)[^\d]{0,8}([\d,]{7,})\s*円")
            if y is not None:
                out["price_man"] = y / 10000.0
    for key, pat in (
        ("area_sqm", r"専有面積[^\d]{0,6}([\d.]+)"),
        ("balcony_sqm", r"バルコニー[^\d]{0,8}([\d.]+)"),
        ("station_walk_min", r"徒歩\s*([\d]+)\s*分"),
        ("monthly_fees_yen", r"管理費[^\d]{0,6}([\d,]+)\s*円"),
        ("repair_reserve_yen", r"修繕積立金[^\d]{0,6}([\d,]+)\s*円"),
        # LABEL PRECEDENCE. A bare 賃料 pattern also matches INSIDE 想定賃料, so an
        # ESTIMATE could populate contractual rent. Contractual labels are
        # explicit; the generic 賃料/家賃 form is accepted only when NOT preceded
        # by 想定/参考/市場/予想.
        ("monthly_rent_yen",
         r"(?:現行賃料|契約賃料|現況賃料|現行家賃|契約家賃)[^\d]{0,6}([\d,]+)\s*円"
         r"|(?<![想定参考市場予])賃料[^\d]{0,6}([\d,]+)\s*円"
         r"|(?<![想定参考市場予])家賃[^\d]{0,6}([\d,]+)\s*円"),
        ("market_rent_yen",
         r"(?:想定賃料|想定家賃|参考賃料|参考家賃|市場賃料|市場家賃|予想賃料)"
         r"[^\d]{0,6}([\d,]+)\s*円"),
        ("total_units", r"総戸数[^\d]{0,4}([\d,]+)"),
        ("unit_floor", r"所在階[^\d]{0,4}([\d]+)"),
    ):
        v = grab(pat)
        if v is not None:
            out[key] = v
    if "monthly_fees_yen" in out and "repair_reserve_yen" in out:
        out["monthly_fees_yen"] = out["monthly_fees_yen"] + out.pop("repair_reserve_yen")
    yv = re.search(r"(?:満室時利回り|想定利回り|表面利回り)[^\d%]{0,6}([\d.]+)\s*[%％]", n)
    if yv:
        out["gross_yield_pct"], out["yield_basis"] = float(yv.group(1)), "full"
    cv = re.search(r"現況利回り[^\d%]{0,6}([\d.]+)\s*[%％]", n)
    if cv:
        out["gross_yield_pct"], out["yield_basis"] = float(cv.group(1)), "current"
    by = _parse_japanese_year(re.search(r"築年月[^\d令平昭]{0,4}([^\s]{2,12})", n).group(1)
                              if re.search(r"築年月[^\d令平昭]{0,4}([^\s]{2,12})", n) else None)
    if by:
        out["building_age_years"] = max(date.today().year - by, 0)
    st_m = re.search(r"構造[^\S\n]{0,4}(RC|SRC|鉄筋コンクリート|鉄骨|木造|軽量鉄骨)", n)
    if st_m:
        out["structure"] = st_m.group(1)
    ad = re.search(r"(?:所在地|住所)[^\S\n]{0,4}([^\n]{6,60})", n)
    if ad:
        out["address_jp"] = ad.group(1).strip()
    nm = re.search(r"(?:物件名|建物名|マンション名)[^\S\n]{0,4}([^\n]{2,50})", n)
    if nm:
        out["building_name"] = nm.group(1).strip(" ■◆●・")
    if re.search(r"賃貸中|入居中", n):
        out["occupancy"] = "rented"
    elif re.search(r"空室", n):
        out["occupancy"] = "vacant"
    # Labelled companies only. Role assignment happens in map_company_roles().
    pairs = []
    for lab in ("管理会社", "賃貸管理会社", "家賃保証会社", "仲介", "媒介",
                "販売会社", "売主", "担当者", "管理組合"):
        cm = re.search(lab + r"[^\S\n]{0,4}([^\n]{2,40})", n)
        if cm:
            pairs.append([lab, cm.group(1).strip()])
    if pairs:
        out["company_roles"] = pairs
    return out

def extract_document_pages(data: bytes, filename: str) -> dict:
    """Return {"pages":[{page,text,image_b64}], "method", "warnings"}.

    Validates size/type/page-count and raises DocumentError with a message the
    user can act on. Encrypted and corrupt files are distinguished, because the
    fix differs (supply the password vs re-export the file)."""
    warnings: list[str] = []
    if not data:
        raise DocumentError(tr("doc_empty"))
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise DocumentError(tr("doc_too_big", size=f"{size_mb:.1f}", limit=MAX_UPLOAD_MB))
    name = (filename or "").lower()
    is_pdf = name.endswith(".pdf") or data[:5] == b"%PDF-"
    if not is_pdf:
        return {"pages": [{"page": 1, "text": "",
                           "image_b64": base64.b64encode(data).decode()}],
                "method": "image", "warnings": warnings}
    ok, err = pdf_available()
    if not ok:
        raise DocumentError(tr("doc_no_pymupdf", err=err[:80]))
    try:
        doc = _open_pdf(data)
    except Exception as e:
        raise DocumentError(tr("doc_corrupt", err=_scrub_secrets(str(e))[:100]))
    try:
        if getattr(doc, "needs_pass", False) or getattr(doc, "is_encrypted", False):
            raise DocumentError(tr("doc_encrypted"))
        total = doc.page_count
        if total > MAX_PDF_PAGES:
            warnings.append(tr("doc_page_cap", total=total, cap=MAX_PDF_PAGES))
        pages = []
        for i in range(min(total, MAX_PDF_PAGES)):
            page = doc.load_page(i)
            # sort=True is essential, not cosmetic. PyMuPDF's default returns
            # text in DRAWING order, which on a spreadsheet-exported 販売図面
            # emits every label first and every value afterwards:
            #     総戸数 バルコニー面積 管理費 修繕積立金 … 119戸 7.39㎡ 8,600 6,200
            # Label-adjacency parsing cannot work on that at all, which is why a
            # page containing every field in plain text extracted almost nothing.
            # Sorting by position restores reading order so labels sit beside
            # their values.
            text = (page.get_text("text", sort=True) or "").strip()
            img_b64, how = "", "text"
            if len(text) < MIN_TEXT_CHARS_PER_PAGE:
                pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
                img_b64 = base64.b64encode(pix.tobytes("png")).decode()
                how = "vision" if not text else "text+vision"
            pages.append({"page": i + 1, "text": text, "image_b64": img_b64,
                          "method": how})
    finally:
        try:
            doc.close()
        except Exception:
            pass
    method = "text" if all(p["text"] and not p["image_b64"] for p in pages) else "mixed"
    return {"pages": pages, "method": method, "warnings": warnings}

def merge_extractions(per_page: list[dict]) -> tuple[dict, list[dict], dict]:
    """Merge page-level extractions; report conflicts instead of picking a winner.

    Non-conflicting fields merge silently. Where two pages disagree, the FIRST
    value is kept and the disagreement is returned for the user to resolve —
    quietly choosing one would hide exactly the discrepancy worth seeing."""
    merged: dict = {}
    provenance: dict = {}
    conflicts: list[dict] = []
    for item in per_page:
        page_no = item.get("page", 0)
        for k, v in (item.get("fields") or {}).items():
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            if k not in merged:
                merged[k] = v
                provenance[k] = {"extracted_value": v,
                                 "source_document": item.get("document"),
                                 "page_number": page_no,
                                 "extraction_method": item.get("method", "vision"),
                                 "confidence": item.get("confidence")}
            elif str(merged[k]).strip() != str(v).strip():
                conflicts.append({"field": k, "kept": merged[k], "other": v,
                                  "page_number": page_no})
                provenance[k].setdefault("conflicts", []).append(
                    {"value": v, "page_number": page_no})
    return merged, conflicts, provenance

def apply_manual_overrides(provenance: dict, reviewed: dict) -> dict:
    """Compare what is being saved against what was extracted.

    A field the user edited keeps its extraction history but is stamped
    manual_override, so an audit row shows both what the document said and what
    the human decided — losing either half makes the record unreviewable."""
    out = dict(provenance or {})
    for field, entry in list(out.items()):
        if field not in reviewed:
            continue
        extracted = entry.get("extracted_value")
        current = reviewed.get(field)
        if extracted is None or str(extracted).strip() == str(current).strip():
            continue
        out[field] = {**entry, "final_value": current,
                      "extraction_method": "manual_override",
                      "original_method": entry.get("extraction_method"),
                      "overridden_at": datetime.now().isoformat(timespec="seconds")}
    return out

# ---------------------------------------------------------------------------
# WS5: source attribution — how the data arrived vs who supplied it
# ---------------------------------------------------------------------------
INGESTION_CHANNELS = ["pasted_text", "agent_pdf", "agent_image", "portal_screenshot",
                      "listing_url", "browser_extension", "manual"]
SOURCE_TYPES = ["real_estate_agency", "property_portal", "seller",
                "management_company", "user_supplied", "unknown"]
AGENCY_ROLES = ["seller", "seller_agent", "buyer_agent", "intermediary",
                "document_issuer", "unknown"]
# Japanese role labels -> which FIELD the company belongs in. Conflating these is
# the specific failure this table exists to prevent: a 管理会社 is not the
# brokerage, and copying it into agency_name misattributes the document.
ROLE_LABEL_MAP = [
    ("賃貸管理会社", "rental_management_company", None),
    ("家賃保証会社", "rent_guarantee_company", None),
    ("管理会社",     "management_company",      None),
    ("管理組合",     "management_association",  None),
    ("仲介",         "agency_name",             "intermediary"),
    ("媒介",         "agency_name",             "intermediary"),
    ("販売会社",     "agency_name",             "seller_agent"),
    ("売主",         "seller_name",             None),
    ("担当者",       "agent_name",              None),
]

def map_company_roles(pairs: list[tuple[str, str]]) -> dict:
    """Assign companies to fields by their printed Japanese role label.

    `pairs` is [(label, company_name), …] as read off the document. Anything
    whose role is unclear goes to `extracted_entities` with role "unknown" and is
    shown for review — it is NEVER silently promoted to agency_name, because an
    agency attributed by guesswork is worse than an unknown one."""
    out: dict = {"extracted_entities": []}
    for label, name in pairs:
        name = (name or "").strip()
        if not name:
            continue
        lab = unicodedata.normalize("NFKC", str(label or "")).strip()
        matched = False
        for needle, field, role in ROLE_LABEL_MAP:   # longest-first ordering above
            if needle in lab:
                out.setdefault(field, name)
                if role:
                    out.setdefault("agency_role", role)
                matched = True
                break
        if not matched:
            out["extracted_entities"].append(
                {"name": name, "label": lab, "role": "unknown"})
    return out

def source_display(row) -> str:
    """Audit 'Source' column: agency > portal > channel label > unknown."""
    get = row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row else d)
    for field in ("agency_name", "portal"):
        val = get(field)
        if val and str(val).lower() not in ("nan", "none", ""):
            return str(val)
    ch = get("ingestion_channel")
    if ch and str(ch).lower() not in ("nan", "none", ""):
        key = f"channel_{ch}"
        return tr(key) if key in TRANSLATIONS["en"] else str(ch)
    return tr("source_unknown")

def _parse_japanese_year(raw) -> int | None:
    """Gregorian year from an MLIT BuildingYear value.

    MLIT returns 「平成10年」 at least as often as 「1998年」. A Gregorian-only
    regex silently dropped the year for a large share of pre-2019 stock — which
    is precisely the stock where building age drives the valuation. Handles all
    five modern eras plus 元年."""
    txt = unicodedata.normalize("NFKC", str(raw or "")).strip()
    if not txt:
        return None
    for era, offset in _ERA_OFFSETS.items():
        if era in txt:
            after = txt.split(era, 1)[1]
            if after.startswith("元"):
                return offset + 1
            mm = re.search(r"\d{1,2}", after)
            if mm:
                n = int(mm.group(0))
                if 1 <= n <= 64:
                    return offset + n
            return None
    mm = re.search(r"(18|19|20)\d{2}", txt)
    if mm:
        yr = int(mm.group(0))
        return yr if 1850 <= yr <= date.today().year + 1 else None
    return None

def _parse_mlit_comparables(payload: dict, city_code: str, year: int, quarter: int) -> list[dict]:
    rows = []
    for d in payload.get("data", []) or []:
        if not isinstance(d, dict): continue
        kind = str(d.get("Type") or "")
        if not any(k in kind for k in ("中古マンション", "宅地")): continue
        # 宅地(土地) is land only and has no building to compare against.
        if "宅地" in kind and "土地と建物" not in kind: continue
        price = safe_float(str(d.get("TradePrice", "")).replace(",", "")); area = safe_float(str(d.get("Area", "")).replace(",", ""))
        if math.isnan(price) or math.isnan(area) or area <= 0: continue
        ppsm = price / area
        if not MLIT_PPSM_BOUNDS[0] <= ppsm <= MLIT_PPSM_BOUNDS[1]: continue
        by = _parse_japanese_year(d.get("BuildingYear") or d.get("ConstructionYear"))
        structure = str(d.get("Structure") or "").strip() or None
        layout = str(d.get("FloorPlan") or "").strip() or None
        ident = f"{city_code}|{year}|{quarter}|{price}|{area}|{by}|{structure}|{layout}"
        rows.append({"comp_id": hashlib.sha256(ident.encode()).hexdigest()[:24], "city_code": city_code,
          "year": year, "quarter": quarter, "property_kind": kind, "floor_plan": layout,
          "area_sqm": area, "building_year": by, "structure": structure, "price_yen": price, "price_per_sqm": ppsm})
    return rows

def save_transaction_comparables(rows: list[dict]) -> None:
    if not rows: return
    cols=["comp_id","city_code","year","quarter","property_kind","floor_plan","area_sqm","building_year","structure","price_yen","price_per_sqm","fetched_at"]
    now=datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        conn.executemany(_upsert_sql("transaction_comparables", cols, ["comp_id"]),
                         [tuple(r.get(c) if c != "fetched_at" else now for c in cols) for r in rows])

# Which stored comparables belong to the same asset class as the subject. A
# 区分マンション must not be valued off detached houses: they trade on different
# ¥/m² entirely, and mixing them biases the median in whichever direction the
# ward's housing mix happens to lean.
CONDO_PTYPES = {"1R Mansion", "1K Apartment", "1DK Apartment", "1LDK Apartment",
                "2K Apartment", "2DK Mansion", "2LDK Mansion", "3DK Mansion",
                "3LDK Mansion", "Office Unit", "Corner Retail Unit"}

# Residential SECTIONAL units only. Offices and retail are excluded even though
# they are also 区分所有: 中古マンション等 transactions are residential and trade at
# entirely different ¥/m², so valuing a shop from flats is the wrong market, not
# merely a thin sample.
RESIDENTIAL_SECTIONAL_PTYPES = CONDO_PTYPES - {"Office Unit", "Corner Retail Unit"}
COMMERCIAL_PTYPES = {"Office Unit", "Corner Retail Unit"}

def _same_class_mask(kinds: pd.Series, subject_ptype: str | None) -> pd.Series:
    """True where a comparable's MLIT 種類 matches the subject's asset class."""
    k = kinds.fillna("").astype(str)
    is_condo = k.str.contains("中古マンション", regex=False)
    if subject_ptype in RESIDENTIAL_SECTIONAL_PTYPES:
        return is_condo
    if subject_ptype in ("Detached House", "Whole Building Apartment"):
        return ~is_condo                      # 宅地(土地と建物)
    if subject_ptype in COMMERCIAL_PTYPES:
        return pd.Series(False, index=kinds.index)   # nothing compatible exists
    return pd.Series(True, index=kinds.index)  # unknown subject: don't filter

MIN_COMPARABLES = 5     # below this, publish nothing rather than a weak number
MAX_COMPARABLES = 40    # strongest N by similarity

def comparable_valuation(city_code, area_sqm, building_age, structure=None,
                         ptype=None) -> dict:
    blank={"available":False,"estimate":float("nan"),"low":float("nan"),"high":float("nan"),
           "n":0,"confidence":"Low","similarity":0.0,"recency_months":None,
           "dispersion_pct":float("nan"),"time_adjusted":False}
    area=safe_float(area_sqm); age=safe_float(building_age)
    if not city_code or math.isnan(area) or area <= 0: return blank
    with get_conn() as conn:
        if isinstance(conn, _PgConnWrapper):
            rows=conn.execute("SELECT * FROM transaction_comparables WHERE city_code=?",(str(city_code),)).fetchall(); df=pd.DataFrame(rows)
        else:
            df=pd.read_sql_query("SELECT * FROM transaction_comparables WHERE city_code=?",conn,params=(str(city_code),))
    if df.empty: return blank
    if ptype in COMMERCIAL_PTYPES:
        return blank | {"unavailable_reason": "no_commercial_comparables"}
    if "property_kind" in df.columns:
        df = df[_same_class_mask(df["property_kind"], ptype)]
        if df.empty: return blank
    df["area_sqm"]=pd.to_numeric(df["area_sqm"],errors="coerce"); df["price_per_sqm"]=pd.to_numeric(df["price_per_sqm"],errors="coerce")
    df=df.dropna(subset=["area_sqm","price_per_sqm"]); df=df[(df.area_sqm>=area*.55)&(df.area_sqm<=area*1.8)]
    if df.empty: return blank
    sim=np.exp(-abs(df.area_sqm-area)/max(area*.35,5.0)); target=date.today().year-int(round(age)) if not math.isnan(age) else None
    if target is not None:
        by=pd.to_numeric(df.building_year,errors="coerce"); sim*=np.exp(-abs(by-target)/18).where(by.notna(),.65)
    if structure: sim*=np.where(df.structure.fillna("").str.contains(str(structure),case=False,regex=False),1.15,.9)
    df["similarity"]=pd.Series(sim,index=df.index).clip(.05,1)
    # ROBUST OUTLIER REJECTION. The ¥/m² sanity bounds are deliberately wide, so
    # a single mispriced or mis-keyed record could still drag the median. Trim by
    # modified z-score (median absolute deviation), which unlike mean/σ is not
    # itself distorted by the outliers it is meant to catch.
    ppsm = pd.to_numeric(df["price_per_sqm"], errors="coerce")
    med_p = float(np.nanmedian(ppsm))
    mad = float(np.nanmedian(np.abs(ppsm - med_p)))
    if mad > 0:
        keep = (0.6745 * np.abs(ppsm - med_p) / mad) <= 3.5
        if keep.sum() >= 5:
            df = df[keep]
    # TIME ADJUSTMENT. Older transactions are restated to today using the ward's
    # own quarterly trend where one exists; without enough history we leave the
    # price untouched rather than invent a drift.
    df = df.assign(adj_factor=1.0)
    trend = get_ward_momentum(city_code)
    if trend:
        annual = trend[0] / 100.0
        now_y = date.today().year; now_q = (date.today().month - 1) // 3 + 1
        yrs = ((now_y - pd.to_numeric(df["year"], errors="coerce"))
               + (now_q - pd.to_numeric(df["quarter"], errors="coerce")) / 4.0)
        df["adj_factor"] = (1.0 + annual).__pow__(yrs.clip(lower=0, upper=8)).fillna(1.0)
    df=df.sort_values("similarity",ascending=False).head(MAX_COMPARABLES)
    if len(df)<MIN_COMPARABLES: return blank|{"n":len(df)}
    vals=np.sort((df.price_per_sqm*df.adj_factor*area).to_numpy(float))
    med=float(np.median(vals)); lo=float(np.quantile(vals,.25)); hi=float(np.quantile(vals,.75))
    ms=float(df.similarity.median())
    # Recency of the comparable set, and how tightly the prices cluster.
    yy=pd.to_numeric(df["year"],errors="coerce"); qq=pd.to_numeric(df["quarter"],errors="coerce")
    months=((date.today().year-yy)*12+(((date.today().month-1)//3+1)-qq)*3)
    recency=float(np.nanmedian(months)) if months.notna().any() else None
    dispersion=float((hi-lo)/med*100.0) if med>0 else float("nan")
    conf="High" if len(df)>=20 and ms>=.55 else "Medium" if len(df)>=10 else "Low"
    return {"available":True,"estimate":med,"low":lo,"high":hi,"n":len(df),
            "confidence":conf,"similarity":ms,"recency_months":recency,
            "dispersion_pct":dispersion,"time_adjusted":bool(trend)}

def financing_analysis(price, noi, down=30.0, rate=2.0, years=30, target_coc=5.0):
    price=safe_float(price); noi=safe_float(noi); dp=safe_float(down,30)/100; r=safe_float(rate,2)/1200; n=max(int(years)*12,1)
    if math.isnan(price) or math.isnan(noi) or price<=0: return {k:float("nan") for k in ("debt","dscr","cashflow","coc","max_price")}
    loan=price*(1-dp); monthly=loan/n if r==0 else loan*r*(1+r)**n/((1+r)**n-1); debt=monthly*12
    cash=price*dp+acquisition_costs(price)["total_costs"]; flow=noi-debt; coc=flow/cash*100 if cash>0 else float("nan")
    # Maximum offer at the target cash-on-cash return. Acquisition costs are NOT
    # strictly proportional to price: brokerage is 3% + a FIXED ¥60,000 (+tax),
    # so treating everything as proportional biased the answer (~0.07pp). Solve
    # properly for P:
    #   noi - P(1-dp)*f = (target/100) * (P*dp + P*b_rate + b_fixed)
    #   => P = (noi - (target/100)*b_fixed) / ((1-dp)*f + (target/100)*(dp+b_rate))
    # where f is annual debt service per yen of loan and b_fixed is the constant.
    f_per_yen = (monthly*12/loan) if loan > 0 else 0.0
    b_fixed = BROKERAGE_FIXED*(1+CONSUMPTION_TAX_PCT/100)
    b_rate = (BROKERAGE_PCT/100)*(1+CONSUMPTION_TAX_PCT/100) + _expense_assumptions()["acq_other_pct"]/100
    t = target_coc/100.0
    denom = (1-dp)*f_per_yen + t*(dp + b_rate)
    maxp = (noi - t*b_fixed)/denom if denom > 0 else float("nan")
    dscr = noi/debt if debt>0 else float("inf")
    # Financing status is reported ALONGSIDE asset quality, never folded into it:
    # a great building with bad bank terms is still a great building, and the two
    # decisions are taken at different times with different information.
    if math.isnan(dscr) or math.isnan(coc):
        status = "not_evaluated"
    elif dscr >= 1.3 and coc >= target_coc:
        status = "acceptable"
    elif dscr >= 1.0:
        status = "tight"
    else:
        status = "unworkable"
    return {"debt":debt,"dscr":dscr,"cashflow":flow,"coc":coc,"max_price":maxp,
            "status":status,"cash_invested":cash}

def recommendation_confidence(row, src, n, hazards, dev, pop, address):
    checks=[src=="MLIT_CITY" and n>=MLIT_MIN_SAMPLES, not bool(row.get("fees_estimated",True)), str(row.get("yield_basis"))!="unknown", bool(address), bool(hazards), bool(dev), bool(pop), str(row.get("zoning"))!="unknown", not math.isnan(far_or_nan(row.get("far_pct")))]
    score=round(sum(checks)/len(checks)*100); return score, "High" if score>=78 else "Medium" if score>=50 else "Low"

def verify_municipality_codes(api_key: str, prefectures: list[str]) -> dict:
    """Check our hard-coded MUNICIPALITY_CODES against MLIT's own 市区町村 list
    (XIT002) and report mismatches.

    Why this exists: every code in MUNICIPALITY_CODES was entered by hand. A
    wrong one fails SILENTLY — the ward simply returns zero samples on every
    refresh forever, or in the worst case quietly returns a different ward's
    data. This turns that invisible failure into an explicit report.

    Returns {pref: {"checked": n, "unknown": [(name, code), ...],
                    "name_mismatch": [(code, ours, theirs), ...], "error": str|None}}
    Never raises.
    """
    report: dict[str, dict] = {}
    for pref in prefectures:
        pref_code = PREF_CODES.get(pref)
        entry = {"checked": 0, "unknown": [], "name_mismatch": [], "error": None}
        if not pref_code:
            entry["error"] = "no prefecture code mapped"
            report[pref] = entry
            continue
        try:
            q = urllib.parse.urlencode({"area": pref_code})
            req = urllib.request.Request(
                f"{MLIT_CITYLIST_ENDPOINT}?{q}",
                headers={"Ocp-Apim-Subscription-Key": api_key,
                         "Accept-Encoding": "gzip"})
            with _urlopen_with_backoff(req, timeout=25) as resp:
                data = resp.read()
                if "gzip" in (resp.headers.get("Content-Encoding") or "").lower() \
                        or data[:2] == b"\x1f\x8b":
                    data = gzip.decompress(data)
            payload = json.loads(data.decode("utf-8", errors="ignore"))
            rows = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(rows, list) or not rows:
                entry["error"] = "empty or unexpected city list response"
                report[pref] = entry
                continue
            # Be liberal about the field names MLIT uses for code/name.
            official: dict[str, str] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("id") or row.get("code") or row.get("city_code") or "").strip()
                name = str(row.get("name") or row.get("city_name") or "").strip()
                if code:
                    official[code] = name
            ours = MUNICIPALITY_CODES.get(pref, {})
            # Only check the Japanese-name keys; romaji aliases are ours alone.
            for name, code in ours.items():
                if name.isascii():
                    continue
                entry["checked"] += 1
                if code not in official:
                    entry["unknown"].append((name, code))
                else:
                    theirs = official[code]
                    # Our keys are compounds like 神戸市中央区 while MLIT may list
                    # just 中央区 — treat containment either way as agreement.
                    if theirs and theirs not in name and name not in theirs:
                        entry["name_mismatch"].append((code, name, theirs))
        except Exception as e:
            entry["error"] = _scrub_secrets(str(e))[:160]
        report[pref] = entry
        time.sleep(MLIT_REQUEST_PAUSE_S)
    return report

def _mlit_fetch_quarter(api_key: str, year: int, quarter: int,
                        city_code: str | None = None,
                        area_code: str | None = None) -> dict:
    """One raw MLIT XIT001 request for a single municipality-quarter (or, for
    backward compatibility, a prefecture-quarter). Isolated so tests can
    monkeypatch the network layer and so callers control batching.

    Narrowing is done with the 5-digit `city` query parameter — the documented
    way to scope XIT001 to one 市区町村 (the official example is
    `?year=2015&quarter=2&city=13102`). When only `area_code` (the 2-digit
    prefecture code) is supplied, the request falls back to prefecture scope.
    The API key travels ONLY in the Ocp-Apim-Subscription-Key header, never the
    URL, so it can't leak into a stringified URLError."""
    query: dict = {"year": year, "quarter": quarter}
    if city_code:
        query["city"] = city_code            # 5-digit 市区町村コード (preferred, hyper-local)
    elif area_code:
        query["area"] = area_code            # 2-digit prefecture code (coarse fallback)
    params = urllib.parse.urlencode(query)
    req = urllib.request.Request(
        f"{MLIT_ENDPOINT}?{params}",
        headers={"Ocp-Apim-Subscription-Key": api_key,
                 "User-Agent": "jp-re-screener/1.0"},
    )
    with _urlopen_with_backoff(req, timeout=25) as resp:
        data = resp.read()
        # The reinfolib manual documents gzip-encoded responses; json-parsing
        # raw gzip bytes yields exactly "Expecting value: line 1 column 1".
        # Handle both the declared header and the gzip magic bytes (belt and
        # braces — some proxies strip Content-Encoding).
        if "gzip" in (resp.headers.get("Content-Encoding") or "").lower() \
                or data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
    text = data.decode("utf-8", errors="ignore")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Don't die mute: surface what the server ACTUALLY sent (HTML error
        # page? empty body? XML?) so the failure is diagnosable from the UI.
        snippet = text.strip()[:80] or "<empty body>"
        raise VisionError(f"non-JSON response from MLIT: {snippet!r}") from e

def fetch_mlit_benchmarks(api_key: str, prefectures: list[str]) -> tuple[dict, dict]:
    """Refresh MUNICIPALITY-level ¥/m² benchmarks from real MLIT transaction data.

    For every target city of each selected prefecture (MLIT_TARGET_CITIES) we
    hit XIT001 once per quarter of last calendar year, scoping each call to that
    single 市区町村 via the 5-digit `city` parameter, then POOL the ¥/m² samples
    across the four quarters. Pooling is deliberate: it reaches MLIT_MIN_SAMPLES
    in thinner suburban wards and is robust to MLIT's 3–6 month publication lag
    (a year-minus-one window is always fully published). Every city and every
    quarter is independently exception-guarded — one bad request never blocks
    the rest. The pooled median is upserted via save_municipality_benchmark,
    from which point the price-efficiency factor scores a Shibuya listing
    against Shibuya sales rather than a Tokyo-wide blend.

    Returns ({city_code: (rate, sample_count)}, {label: error_message}); error
    labels are the municipality's display name so the console stays readable."""
    results: dict[str, tuple[float, int]] = {}
    errors: dict[str, str] = {}
    dead_streak = 0
    year = date.today().year - 1
    for pref in prefectures:
        pref_code = PREF_CODES.get(pref)
        if not pref_code:
            errors[pref_name(pref)] = "no JIS code mapped"
            continue
        # Dedupe in case a target list ever repeats a code, preserving order.
        target_cities = list(dict.fromkeys(MLIT_TARGET_CITIES.get(pref, [])))
        if not target_cities:
            errors[pref_name(pref)] = "no target municipalities configured"
            continue
        for city_code in target_cities:
            ppsms: list[float] = []
            last_err: str | None = None
            for quarter in (1, 2, 3, 4):
                try:
                    payload = _mlit_fetch_quarter(api_key, year, quarter, city_code=city_code)
                    q_samples = _parse_mlit_payload(payload)
                    ppsms += q_samples
                    save_transaction_comparables(_parse_mlit_comparables(payload, city_code, year, quarter))
                    # Momentum raw material: store the QUARTERLY median too
                    # (same API call — zero extra requests), gated so a thin
                    # quarter can't inject noise into the trend.
                    if len(q_samples) >= MLIT_MIN_QTR_SAMPLES:
                        save_quarterly_median(city_code, year, quarter,
                                              float(np.median(q_samples)), len(q_samples))
                except urllib.error.HTTPError as e:
                    # Fail FAST on auth errors: a bad/expired key fails every one
                    # of the ~dozens of city×quarter calls identically, so abort
                    # the whole refresh immediately rather than sleeping through
                    # the full storm to hand back a wall of 401s.
                    if e.code in (401, 403):
                        return results, {tr("mlit_auth_error_label"): tr("mlit_auth_error")}
                    last_err = _scrub_secrets(str(e))[:120]
                except Exception as e:   # network, schema drift — keep going
                    last_err = _scrub_secrets(str(e))[:120]
                time.sleep(MLIT_REQUEST_PAUSE_S)   # be polite across the many city/quarter calls
            label = municipality_label(city_code)
            if len(ppsms) < MLIT_MIN_SAMPLES:
                errors[label] = (f"only {len(ppsms)} usable samples"
                                 + (f" (last error: {last_err})" if last_err else ""))
                # Circuit breaker: when several wards in a row return NOTHING
                # with errors, the API itself is down/misbehaving for this
                # session — abort instead of grinding through the remaining
                # dozens of city×quarter calls (mirrors the 401/403 fail-fast).
                if len(ppsms) == 0 and last_err:
                    dead_streak += 1
                    if dead_streak >= 3:
                        errors[tr("mlit_circuit_label")] = tr("mlit_circuit_msg")
                        return results, errors
                else:
                    dead_streak = 0
                continue
            dead_streak = 0
            rate = float(np.median(ppsms))
            save_municipality_benchmark(city_code, pref_code, rate, len(ppsms))
            results[city_code] = (rate, len(ppsms))
    return results, errors

# ----------------------------------------------------------------------------
# Unified Data Schema & Mock Scraper
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Portal adapter framework
# ----------------------------------------------------------------------------
# One thin adapter per data source behind a shared contract: each returns rows
# in the unified schema, so analyze_properties, the factor snapshots, and the
# walk-forward loop never change when a source is added, breaks, or is swapped.
#
# Unified schema (one row per listing):
#     listing_id, url, portal, title, ptype, prefecture, station_min,
#     building_age, area_sqm, price_yen, gross_yield, zoning, far_pct, status

class PortalError(Exception):
    """Raised by an adapter when its source can't be fetched/parsed; the caller
    falls back to the mock so one broken portal never kills the whole scan."""

# ----------------------------------------------------------------------------
# Quantitative Scoring Engine
# ----------------------------------------------------------------------------
def _expense_assumptions() -> dict[str, float]:
    """Live expense assumptions: session overrides from the KPI Weights tab if
    the user has tuned them, else the module defaults."""
    return {
        "fee_per_sqm":     float(st.session_state.get("exp_fee_per_sqm", FEE_PER_SQM_MONTH)),
        "owner_maint_pct": float(st.session_state.get("exp_owner_maint_pct", OWNER_MAINT_PCT)),
        "vacancy_pct":     float(st.session_state.get("exp_vacancy_pct", VACANCY_PCT)),
        "mgmt_pct":        float(st.session_state.get("exp_mgmt_pct", MGMT_PCT)),
        "tax_pct":         float(st.session_state.get("exp_tax_pct", TAX_PCT_OF_PRICE)),
        "acq_other_pct":   float(st.session_state.get("exp_acq_other_pct", ACQ_OTHER_PCT)),
    }

def get_small_unit_penalty() -> float:
    return float(st.session_state.get("small_unit_penalty", SMALL_UNIT_PENALTY))

def acquisition_costs(price_yen) -> dict:
    """Closing costs (諸費用) and total invested capital.

    Brokerage is computed exactly (3% + ¥60,000 + consumption tax) because its
    fixed component makes cheap units proportionally worse; everything else
    (不動産取得税, 登録免許税, 司法書士, 印紙税, insurance, tax settlement) is a
    single tunable percentage. Returns NaNs for an unusable price."""
    a = _expense_assumptions()
    price = safe_float(price_yen)
    if math.isnan(price) or price <= 0:
        return {"brokerage": float("nan"), "other": float("nan"),
                "total_costs": float("nan"), "total_invested": float("nan"),
                "costs_pct": float("nan")}
    brokerage = (price * BROKERAGE_PCT / 100.0 + BROKERAGE_FIXED) * (1 + CONSUMPTION_TAX_PCT / 100.0)
    other = price * a["acq_other_pct"] / 100.0
    total = brokerage + other
    return {"brokerage": brokerage, "other": other, "total_costs": total,
            "total_invested": price + total, "costs_pct": total / price * 100.0}

# ---------------------------------------------------------------------------
# WS6: evidence-weighted size resilience
# ---------------------------------------------------------------------------
# STATUS: EVIDENCE-WEIGHTED BY DEFAULT, WITH SAFE FALLBACK MODES.
# Exactly one size treatment reaches the production score: legacy_penalty,
# descriptive_only, or evidence_weighted. The evidence-weighted adjustment is
# continuous, capped, and deliberately excluded from system_weights and the
# walk-forward learner until outcome back-testing supports calibration.
#
# It offers a transparent estimate of whether a unit stays functional, rentable,
# financeable and saleable over a long hold. Two things are deliberately separated:
#   * a GENERIC PRIOR from floor area alone, and
#   * LOCAL EVIDENCE from comparable transactions,
# blended by how strong that evidence actually is. Absent evidence therefore
# lowers CONFIDENCE and pulls toward the prior — it does not invent a penalty.
#
# The anchors are NOT a claim that occupants will want exactly this many m² in
# fifteen years. They encode today's financing and resale structure: sub-20m² is
# hard to finance, ~25m² is the ward ワンルーム ordinance floor, and above ~35m²
# the curve PLATEAUS because bigger is not automatically a better investment.
SIZE_PRIOR_ANCHORS = [(0.0, 25.0), (15.0, 30.0), (18.0, 45.0), (22.0, 60.0),
                      (25.0, 70.0), (30.0, 78.0), (35.0, 82.0), (40.0, 85.0),
                      (50.0, 85.0), (100.0, 82.0)]
SIZE_SUBWEIGHTS = {"functional": 0.35, "rental": 0.35, "exit": 0.30}
EVIDENCE_STRENGTH_CAP = 0.80
# How floor area is allowed to move the production score.
#   legacy_penalty    — the original binary deduction (full under 20m², half
#                       under 25m²). Kept for comparison and migration testing.
#   descriptive_only  — no size effect on the score at all; the assessment is
#                       shown alongside it.
#   evidence_weighted — DEFAULT. The legacy deduction is NOT applied; instead a
#                       continuous, bounded adjustment derived from the
#                       resilience assessment, clamped to ±SIZE_ADJ_CAP so a
#                       still-uncalibrated model can never dominate a score.
SIZE_MODEL_MODES = ("legacy_penalty", "descriptive_only", "evidence_weighted")
SIZE_MODEL_MODE_DEFAULT = "evidence_weighted"
SIZE_ADJ_CAP = 6.0          # maximum absolute points from the size model
SIZE_ADJ_NEUTRAL = 60.0     # resilience value treated as "no adjustment"

def get_size_model_mode() -> str:
    mode = st.session_state.get("size_model_mode",
                                _get_secret("SIZE_MODEL_MODE", SIZE_MODEL_MODE_DEFAULT))
    return mode if mode in SIZE_MODEL_MODES else SIZE_MODEL_MODE_DEFAULT

def size_score_adjustment(res: dict) -> float:
    """Points the size model contributes to the production score.

    Scaled by evidence strength AND clamped to ±SIZE_ADJ_CAP: thin evidence
    pulls toward the neutral prior rather than asserting a large correction, and
    even maximal evidence cannot swing a score by more than the cap. Returns 0.0
    when the model does not apply, so non-sectional stock is untouched."""
    if not res or not res.get("applicable"):
        return 0.0
    final = safe_float(res.get("final"))
    if math.isnan(final):
        return 0.0
    raw = (final - SIZE_ADJ_NEUTRAL) / 40.0 * SIZE_ADJ_CAP
    # Confidence gate: an unevidenced assessment is damped toward zero.
    conf = 0.45 + 0.55 * float(res.get("evidence", 0.0) or 0.0)
    return max(-SIZE_ADJ_CAP, min(SIZE_ADJ_CAP, raw * conf))   # until back-testing calibrates this, never let
                               # local evidence fully displace the prior
# Only sectional residential units get the full model; everything else is N/A.
SIZE_MODEL_PTYPES = CONDO_PTYPES - {"Office Unit", "Corner Retail Unit"}

def size_prior(area_sqm) -> float:
    """Continuous piecewise-linear prior from floor area. No categorical jumps:
    14.9 / 15.0 / 15.1 differ by a hair, not by a cliff."""
    a = safe_float(area_sqm)
    if math.isnan(a) or a <= 0:
        return float("nan")
    pts = SIZE_PRIOR_ANCHORS
    if a <= pts[0][0]:
        return pts[0][1]
    if a >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= a <= x1:
            t = 0.0 if x1 == x0 else (a - x0) / (x1 - x0)
            return round(y0 + t * (y1 - y0), 2)
    return pts[-1][1]

def local_evidence_strength(sample_count, median_similarity, recency_months) -> float:
    """How much the local comparable set deserves to override the prior."""
    sample_strength = min(max(safe_float(sample_count, 0.0), 0.0) / 20.0, 1.0)
    similarity_strength = max(0.0, min(safe_float(median_similarity, 0.0), 1.0))
    recency_strength = max(0.0, 1.0 - max(safe_float(recency_months, 99.0), 0.0) / 36.0)
    return (0.40 * sample_strength + 0.40 * similarity_strength
            + 0.20 * recency_strength)

def functional_adequacy(features: dict | None, area_sqm) -> tuple[float, list[str]]:
    """Does the unit work as a home? Missing features are NEUTRAL, not negative —
    an unanswered question is not a defect. Returns (score, missing_fields)."""
    f = features or {}
    checks = {
        "separate_bath_toilet": 12.0, "indoor_laundry": 12.0, "storage": 8.0,
        "sleep_living_separation": 10.0, "kitchen_functional": 8.0,
        "balcony": 6.0, "elevator": 6.0, "step_free": 4.0, "security": 4.0,
        "delivery_box": 4.0,
    }
    base, missing, earned, available = 50.0, [], 0.0, 0.0
    for key, weight in checks.items():
        val = f.get(key)
        if val is None:
            missing.append(key)
            continue
        available += weight
        if val:
            earned += weight
    if available <= 0:
        # Nothing known: fall back to what area alone implies, flagged as missing.
        return size_prior(area_sqm), missing
    ratio = earned / available
    return round(base + (ratio - 0.5) * 2 * 45.0, 1), missing

def _rent_evidence_from_row(r) -> dict:
    """Structured rental evidence from whatever the row actually carries.

    Absent inputs stay absent — they lower confidence rather than being filled
    with optimistic defaults."""
    get = r.get if hasattr(r, "get") else (lambda k, d=None: r[k] if k in r else d)
    rent = safe_float(get("monthly_rent_yen"))
    area = safe_float(get("area_sqm"))
    # A subject lease proves what THIS unit earns today. It says nothing about
    # what the local market repeats, so it is contract_verified and nothing more.
    # "achieved_local" is reserved for a SEPARATE observation of comparable local
    # rents, which the app cannot yet source — claiming it here would manufacture
    # confidence the data does not support.
    has_contract = not math.isnan(rent) and rent > 0
    declared = str(get("rent_source_kind") or "").strip().lower()
    ev = {"contract_verified": has_contract,
          "source": declared if declared in ("achieved_local", "agent_supported",
                                             "agent_unsupported") else
                    ("contract_verified" if has_contract else "")}
    if not math.isnan(rent) and not math.isnan(area) and area > 0:
        ev["subject_rent_per_sqm"] = rent / area
    for key, field in (("local_median_rent_per_sqm", "local_median_rent_per_sqm"),
                       ("rental_comparable_count", "rental_comparable_count"),
                       ("median_similarity", "rent_median_similarity"),
                       ("recency_months", "rent_recency_months"),
                       ("median_listing_duration_days", "median_listing_duration_days")):
        v = safe_float(get(field))
        if not math.isnan(v):
            ev[key] = v
    return ev

def size_resilience(area_sqm, ptype, city_code=None, building_age=None,
                    structure=None, features=None, rent_evidence=None) -> dict:
    """Blend the generic prior with local comparable evidence.

    Returns every intermediate value so the UI can show the derivation. For
    non-sectional stock (whole buildings, houses, offices, retail) this returns
    applicable=False rather than a misleading number."""
    out = {"applicable": False, "final": float("nan"), "prior": float("nan"),
           "local": float("nan"), "evidence": 0.0, "functional": float("nan"),
           "rental": float("nan"), "exit": float("nan"), "n": 0,
           "similarity": 0.0, "recency_months": None, "missing": [],
           "drivers_pos": [], "drivers_neg": [], "rent_weight": 0.0,
           "rent_locally_benchmarked": False, "rent_source_kind": "none"}
    if ptype not in SIZE_MODEL_PTYPES:
        return out
    prior = size_prior(area_sqm)
    if math.isnan(prior):
        return out
    out.update({"applicable": True, "prior": prior})

    func, missing = functional_adequacy(features, area_sqm)
    out["functional"] = func
    out["missing"] = missing

    # Exit-market breadth from COMPLETED transactions of the same class/size.
    comp = comparable_valuation(city_code, area_sqm, building_age,
                                structure=structure, ptype=ptype)
    n, sim = int(comp.get("n", 0)), float(comp.get("similarity", 0.0) or 0.0)
    recency = comp.get("recency_months")
    out.update({"n": n, "similarity": sim, "recency_months": recency})
    # Breadth: how many close-size completed sales the ward actually produces.
    # Exit-market breadth from COMPLETED same-category transactions: how many,
    # how similar, how recent, and how tightly priced. Wide dispersion means the
    # market is thin or heterogeneous, which is itself an exit risk.
    if n:
        exit_score = 35.0 + min(n, 25) * 1.8
        exit_score += 10.0 * max(0.0, min(1.0, sim))
        disp = safe_float(comp.get("dispersion_pct"))
        if not math.isnan(disp):
            exit_score -= max(0.0, (disp - 25.0) / 25.0) * 12.0
        rec = safe_float(comp.get("recency_months"))
        if not math.isnan(rec):
            exit_score -= max(0.0, (rec - 12.0) / 24.0) * 8.0
        exit_score = max(20.0, min(100.0, exit_score))
    else:
        exit_score = float("nan")     # unknown, never "rejected"
    out["exit"] = exit_score

    # Rental resilience. Two SEPARATE questions, previously conflated:
    #   1. how much do we trust the rent figure?   -> rent_w (evidence strength)
    #   2. is that rent locally REPEATABLE?        -> rental (the score)
    # A verified contract answers (1) strongly but says nothing about (2): a lease
    # signed well above the local median is a re-letting risk, not a strength. So
    # a verified contract no longer grants a flat 80.
    rent_ev = rent_evidence or {}
    src_kind = str(rent_ev.get("source") or "").lower()
    verified = bool(rent_ev.get("contract_verified"))
    # Evidence strength by SOURCE. A verified contract is the strongest statement
    # about this unit, but see below: without local comparables it still cannot
    # support a repeatability judgement, so its weight is trimmed there.
    rent_w = (0.80 if src_kind == "achieved_local" else
              1.00 if verified else
              0.45 if src_kind == "agent_supported" else
              0.20 if src_kind == "agent_unsupported" else 0.0)
    subj = safe_float(rent_ev.get("subject_rent_per_sqm"))
    loc = safe_float(rent_ev.get("local_median_rent_per_sqm"))
    n_rent = safe_float(rent_ev.get("rental_comparable_count"), 0.0)
    sim_rent = safe_float(rent_ev.get("median_similarity"), 0.0)
    recency_rent = safe_float(rent_ev.get("recency_months"), 99.0)
    listing_days = safe_float(rent_ev.get("median_listing_duration_days"))
    if rent_w <= 0:
        rental = float("nan")
    elif not math.isnan(subj) and not math.isnan(loc) and loc > 0 and n_rent >= 3:
        # Repeatability: how far the subject rent sits above/below the local
        # median. At or below median is comfortably re-lettable; well above it
        # means the current tenant is carrying the yield.
        ratio = subj / loc
        rental = 78.0 - max(0.0, (ratio - 1.0)) * 120.0 + max(0.0, (1.0 - ratio)) * 25.0
        if not math.isnan(listing_days):
            # Long local void periods weaken repeatability regardless of price.
            rental -= max(0.0, (listing_days - 30.0) / 60.0) * 10.0
        rental = max(20.0, min(95.0, rental))
        # Thin/old/dissimilar local rent data pulls the judgement toward neutral.
        loc_conf = min(1.0, n_rent / 12.0) * (0.5 + 0.5 * min(1.0, max(0.0, sim_rent))) \
                   * max(0.25, 1.0 - max(0.0, recency_rent) / 36.0)
        rental = 60.0 + (rental - 60.0) * loc_conf
    else:
        # Rent figure trusted, but no local basis to judge repeatability: stay
        # near neutral rather than rewarding the mere existence of a lease.
        rental = 62.0 if verified else 58.0
        # Locally UNBENCHMARKED: the rent is known, its repeatability is not.
        # Halve the weight so a contract alone cannot drive the assessment.
        rent_w *= 0.5
        out["rent_locally_benchmarked"] = False
    out["rental"] = round(rental, 1) if not math.isnan(rental) else float("nan")
    out["rent_weight"] = round(rent_w, 2)
    out.setdefault("rent_locally_benchmarked", True)
    out["rent_source_kind"] = src_kind or ("contract_verified" if verified else "none")

    parts, weights = [], []
    for key, val in (("functional", func), ("rental", rental), ("exit", exit_score)):
        if not math.isnan(safe_float(val, float("nan"))):
            parts.append(val * SIZE_SUBWEIGHTS[key]); weights.append(SIZE_SUBWEIGHTS[key])
    local = (sum(parts) / sum(weights)) if weights else float("nan")
    out["local"] = round(local, 1) if not math.isnan(local) else float("nan")

    strength = local_evidence_strength(n, sim, recency if recency is not None else 99.0)
    # Rental evidence quality and the presence of functional data both temper how
    # far we trust the local picture.
    strength *= (0.55 + 0.45 * rent_w) if rent_w else 0.55
    strength = min(strength, EVIDENCE_STRENGTH_CAP)
    if math.isnan(local):
        strength = 0.0
    out["evidence"] = round(strength, 3)
    final = prior * (1 - strength) + (local if not math.isnan(local) else prior) * strength
    out["final"] = round(final, 1)

    for label, val in (("functional adequacy", func), ("exit-market breadth", exit_score),
                       ("rental resilience", rental)):
        if math.isnan(safe_float(val, float("nan"))):
            continue
        (out["drivers_pos"] if val >= prior else out["drivers_neg"]).append(
            f"{label} {val:.0f}")
    return out

def area_liquidity(area_sqm, ptype=None) -> dict:
    """Exit-liquidity classification by floor area.

    Returns {"tier", "penalty", "owner_occ_eligible"} where tier is:
      "sub_loan"  — under AREA_LOAN_FLOOR: investment loans commonly refused,
                    cash-buyer-only exit. Penalised.
      "caution"   — under AREA_LENDER_CAUTION: some lenders decline; ward
                    ワンルーム ordinances cluster here. Half penalty.
      "investor"  — normal investor stock; no penalty.
      "broad"     — at/above AREA_OWNER_OCC: owner-occupiers can use the
                    住宅ローン控除, so the exit pool is widest.
      "unknown"   — no area supplied.
    Whole buildings and houses are exempt: `area_sqm` there is the whole
    structure, so per-unit thresholds are meaningless.
    """
    area = safe_float(area_sqm)
    if math.isnan(area) or area <= 0:
        return {"tier": "unknown", "penalty": 0.0, "owner_occ_eligible": False}
    if ptype in OWNER_MAINTAINED_PTYPES:
        return {"tier": "investor", "penalty": 0.0, "owner_occ_eligible": area >= AREA_OWNER_OCC}
    pen = get_small_unit_penalty()
    if area < AREA_LOAN_FLOOR:
        return {"tier": "sub_loan", "penalty": pen, "owner_occ_eligible": False}
    if area < AREA_LENDER_CAUTION:
        return {"tier": "caution", "penalty": pen * 0.5, "owner_occ_eligible": False}
    if area >= AREA_OWNER_OCC:
        return {"tier": "broad", "penalty": 0.0, "owner_occ_eligible": True}
    return {"tier": "investor", "penalty": 0.0, "owner_occ_eligible": False}

def get_old_seismic_penalty() -> float:
    return float(st.session_state.get("seismic_penalty", OLD_SEISMIC_PENALTY))

def compute_net_yield(price_yen, gross_yield_pct, area_sqm, ptype,
                      monthly_fees_yen=None, yield_basis: str = YIELD_BASIS_DEFAULT,
                      monthly_rent_yen=None, market_rent_yen=None) -> dict:
    """Net (NOI) yield plus the full cost derivation.

        net = (rent × (1 − vacancy) × (1 − mgmt) − fees − tax) ÷ price

    `yield_basis` matters and is not cosmetic. 満室時/想定/表面 figures assume
    FULL occupancy, so the vacancy allowance is subtracted. A 現況 figure is the
    building's CURRENT actual yield and already embeds its real occupancy —
    deducting vacancy again would double-count it, so we don't. Getting this
    wrong understates a 現況-quoted listing by roughly the vacancy rate.

    Also returns acquisition costs (諸費用) and `net_yield_on_invested`: yield on
    price + closing costs, i.e. on capital actually deployed, which is always
    lower than the yield-on-price figure portals advertise.

    Returns every intermediate figure so the UI can show the derivation instead
    of an unexplained number, plus `fees_estimated` so an estimate is never
    silently passed off as the listing's real figure. NEVER raises: bad or
    missing inputs give net_yield_pct = NaN, which callers must treat as
    'unknown' — never as zero.
    """
    a = _expense_assumptions()
    price, gy, area = safe_float(price_yen), safe_float(gross_yield_pct), safe_float(area_sqm)
    out = {"net_yield_pct": float("nan"), "annual_rent": float("nan"),
           "effective_rent": float("nan"), "annual_fees": float("nan"),
           "annual_tax": float("nan"), "noi": float("nan"), "fees_estimated": False,
           "vacancy_applied": 0.0, "yield_basis": yield_basis,
           "rent_source": "gross_yield", "contractual_gross_pct": float("nan"),
           "reported_gross_pct": float("nan"), "gross_gap_pp": float("nan"),
           "rent_conflict": False,
           "acq_costs": float("nan"), "total_invested": float("nan"),
           "net_yield_on_invested": float("nan")}
    contract = safe_float(monthly_rent_yen)
    has_contract = not math.isnan(contract) and contract > 0
    if math.isnan(price) or price <= 0 or (not has_contract and (math.isnan(gy) or gy <= 0)):
        return out                      # neither contract nor gross yield can support rent
    # BASE RENT PRECEDENCE. A verified contractual rent is what the owner
    # actually receives, so it drives the base case. The advertised gross yield
    # is a DERIVED figure and only fills in when no contract rent is known.
    # market_rent_yen is accepted so callers can pass it without risk — it is
    # deliberately never used here; upside belongs in its own scenario.
    if has_contract:
        annual_rent = contract * 12.0
        out["rent_source"] = "contract"
        # Where both exist, disagreement is worth surfacing rather than hiding:
        # a listing advertising a yield its own lease does not support is exactly
        # the discrepancy to catch before offering.
        implied = annual_rent / price * 100.0
        out["contractual_gross_pct"] = round(implied, 2)
        if not math.isnan(gy) and gy > 0:
            out["reported_gross_pct"] = round(gy, 2)
            out["gross_gap_pp"] = round(implied - gy, 2)
            if abs(implied - gy) >= 0.5:
                out["rent_conflict"] = True
    else:
        annual_rent = price * gy / 100.0
        out["rent_source"] = "gross_yield"
        out["reported_gross_pct"] = round(gy, 2)
    supplied = safe_float(monthly_fees_yen)
    has_supplied = not math.isnan(supplied) and supplied > 0
    if has_supplied:
        annual_fees, estimated = supplied * 12.0, False
    elif ptype in OWNER_MAINTAINED_PTYPES:
        # No 管理組合 — the owner funds maintenance/reserves out of rent.
        annual_fees, estimated = annual_rent * a["owner_maint_pct"] / 100.0, True
    else:
        # Condo-type: estimate 管理費+修繕積立金 from floor area.
        per_month = a["fee_per_sqm"] * area if (not math.isnan(area) and area > 0) else 0.0
        annual_fees, estimated = per_month * 12.0, True
    # THE BASIS FIX: only deduct vacancy from a full-occupancy figure. A 現況
    # yield already reflects actual occupancy.
    vac = 0.0 if yield_basis == "current" else a["vacancy_pct"]
    effective_rent = annual_rent * (1 - vac / 100.0) * (1 - a["mgmt_pct"] / 100.0)
    annual_tax = price * a["tax_pct"] / 100.0
    noi = effective_rent - annual_fees - annual_tax
    acq = acquisition_costs(price)
    invested = acq["total_invested"]
    out.update({"net_yield_pct": round(noi / price * 100.0, 2),
                "annual_rent": annual_rent, "effective_rent": effective_rent,
                "annual_fees": annual_fees, "annual_tax": annual_tax,
                "noi": noi, "fees_estimated": estimated,
                "vacancy_applied": vac, "yield_basis": yield_basis,
                "acq_costs": acq["total_costs"], "total_invested": invested,
                "net_yield_on_invested": (round(noi / invested * 100.0, 2)
                                          if invested and not math.isnan(invested) else float("nan"))})
    return out

def seismic_class(building_age, ref_year: int | None = None) -> tuple[str, int | None]:
    """Classify a building against 新耐震基準. Returns (class, completion_year).

      "old"  — completed 1981 or earlier: 旧耐震.
      "grey" — 1982–83: probably 新耐震, but the building permit may predate
               1981-06-01, so the 検査済証 is worth checking.
      "new"  — 1984 onward.
      "unknown" — age not supplied (never guessed as safe).

    Building age is the only input the app reliably has, so the completion year
    is derived (reference year − age) and is therefore ±1 year approximate —
    which is exactly why 1982–83 is flagged grey rather than assumed compliant.
    """
    age = safe_float(building_age)
    if math.isnan(age) or age < 0:
        return "unknown", None
    year = (ref_year or date.today().year) - int(round(age))
    if year >= SEISMIC_NEW_FROM:
        return "new", year
    if year >= SEISMIC_GREY_FROM:
        return "grey", year
    return "old", year

def yield_basis_name(key: str) -> str:
    return tr(f"ybasis_{key}")

def area_tier_name(key: str | None) -> str:
    """Human label for an exit-liquidity tier. Guarded: tr() echoes an unknown
    key back verbatim, so a missing/NaN tier used to surface in the UI as the
    literal string "atier_nan"."""
    k = str(key or "").strip().lower()
    if k in ("", "nan", "none", "unknown"):
        return tr("rv_unchecked")
    return tr(f"atier_{k}") if f"atier_{k}" in TRANSLATIONS["en"] else tr("rv_unchecked")

def hazard_flags_label(flags, short: bool = False) -> str:
    """Decode stored hazard flags into readable text.

    Persisted compactly as e.g. "flood:rank4,landslide:class2" — good for the
    database, meaningless on screen. Returns "Flood inundation zone (depth
    5–10m), Landslide hazard zone (SPECIAL warning zone 特別警戒区域)"."""
    raw = str(flags or "").strip()
    if not raw or raw.lower() in ("nan", "none"):
        return ""
    sep = "、" if get_lang() == "ja" else ", "
    parts = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        hz, _, detail = token.partition(":")
        if short:
            # Chips are small: a two-word name plus the bare depth/class.
            name = tr(f"hzshort_{hz}") if f"hzshort_{hz}" in TRANSLATIONS["en"] else hz
            if detail and f"hzshort_{detail}" in TRANSLATIONS["en"]:
                name += f" {tr('hzshort_' + detail)}"
        else:
            name = tr(f"hazard_{hz}") if f"hazard_{hz}" in TRANSLATIONS["en"] else hz
            if detail and f"hzsev_{detail}" in TRANSLATIONS["en"]:
                name += f" ({tr('hzsev_' + detail)})"
        parts.append(name)
    return sep.join(parts)

def dev_flags_label(flags, short: bool = False) -> str:
    """Decode stored development flags ("road,chiku,activity") into readable
    text rather than showing the internal keys."""
    raw = str(flags or "").strip()
    if not raw or raw.lower() in ("nan", "none"):
        return ""
    sep = "、" if get_lang() == "ja" else ", "
    parts = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            key = f"devshort_{token}" if short else f"dev_{token}"
            parts.append(tr(key) if key in TRANSLATIONS["en"] else token)
    return sep.join(parts)

# A TOTAL-population field is a recognised stem followed by the year, with at
# most an underscore between. The anchoring is the whole point: 国土数値情報 mesh
# data puts five-year AGE BRACKETS right next to the totals (PT01_2020 = ages
# 0-4, PTA_2020 = a grouped cohort), and a loose prefix test cannot tell them
# apart. An earlier version here normalised keys by STRIPPING underscores, which
# destroyed exactly the boundary that separates PT0_2020 (total) from PT01_2020
# (a cohort) — so a total in one year could be compared against a cohort in
# another and report a ~90% "collapse" for a healthy ward.
_POP_TOTAL_RE = re.compile(r"^(PT0|PTN|POPT|POP|JINKO)_?(\d{4})$", re.IGNORECASE)

def _feature_population_series(props: dict) -> dict[str, dict[int, float]]:
    """Pull {stem: {year: total_population}} out of one 将来推計人口 mesh feature.

    Returns series keyed BY FIELD STEM rather than flattened, so the caller can
    insist on comparing like with like across years. Age brackets are excluded
    structurally by the anchored regex, not by ordering luck."""
    by_stem: dict[str, dict[int, float]] = {}
    for k, v in (props or {}).items():
        if not isinstance(k, str):
            continue
        mm = _POP_TOTAL_RE.match(k.strip())
        if not mm:
            continue
        stem, year = mm.group(1).upper(), int(mm.group(2))
        if not (2000 <= year <= 2070):
            continue
        val = safe_float(v)
        if not math.isnan(val) and val >= 0:
            by_stem.setdefault(stem, {})[year] = val
    return by_stem

def check_population_outlook(lat: float | None, lon: float | None) -> dict:
    """Projected population change in the listing's catchment (XKT013).

    Sums every 500m mesh in the fetched tile per projection year, then compares
    the nearest available base year against the year closest to the resale
    horizon. Summing the whole ~1km tile rather than the single mesh the building
    sits in is deliberate: resale buyers come from the surrounding area, and an
    aggregate is far less sensitive to which side of a mesh boundary the point
    happens to land on.

    Returns {"pct_change", "from_year", "to_year", "base_pop", "horizon_pop",
    "band"} or {} when unavailable. Same contract as the other MLIT checks:
    needs a key + coordinates, caches only complete results, NEVER raises.
    """
    if lat is None or lon is None:
        return {}
    api_key = _get_secret("MLIT_API_KEY")
    if not api_key:
        return {}
    horizon = int(st.session_state.get("resale_horizon", RESALE_HORIZON_YEARS))
    cache = st.session_state.setdefault("_pop_cache", {})
    ckey = f"{round(lat,5)},{round(lon,5)}|{horizon}"
    if ckey in cache:
        return cache[ckey]
    x, y = _latlon_to_tile(lat, lon, HAZARD_TILE_Z)
    try:
        payload = _fetch_hazard_tile(POP_ENDPOINT, api_key, HAZARD_TILE_Z, x, y)
        feats = payload.get("features", []) if isinstance(payload, dict) else []
    except Exception:
        return {}          # transient -> unknown, and NOT cached (retry later)
    # Aggregate per FIELD STEM so every year in a comparison comes from the same
    # field. Mixing stems is what produced the "-92%" artifact: a total in the
    # base year measured against an age cohort in the horizon year.
    stems: dict[str, dict[int, float]] = {}
    for f in feats:
        for stem, series in _feature_population_series(f.get("properties", {})).items():
            tgt = stems.setdefault(stem, {})
            for yr, val in series.items():
                tgt[yr] = tgt.get(yr, 0.0) + val
    # Prefer the stem covering the most years, then the largest base population
    # (the genuine total always dominates any single cohort).
    best = None
    for stem, totals in stems.items():
        years = sorted(y_ for y_, v in totals.items() if v > 0)
        if len(years) < 2:
            continue
        cand = (len(years), totals[years[0]], stem, totals, years)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return {}
    _, _, stem, totals, years = best
    base_year = years[0]
    target = base_year + horizon
    to_year = min(years[1:], key=lambda y_: abs(y_ - target))
    base_pop, horizon_pop = totals[base_year], totals[to_year]
    if base_pop <= 0:
        return {}
    pct = (horizon_pop / base_pop - 1.0) * 100.0
    abs_loss = base_pop - horizon_pop
    out = {"pct_change": round(pct, 1), "from_year": base_year, "to_year": to_year,
           "base_pop": base_pop, "horizon_pop": horizon_pop, "field": stem,
           "abs_change": round(-abs_loss, 0)}
    # GUARD 1 — too few people for a percentage to carry information. A tile that
    # is mostly river, park, rail yard or pure commercial floorspace lands here.
    if base_pop < POP_MIN_BASE:
        out.update({"band": "low_base", "pct_change": None})
        cache[ckey] = out
        return out
    # GUARD 2 — physically implausible: a parsing or coverage artifact, not a
    # forecast. Surfaced honestly instead of being scored.
    if abs(pct) > RESALE_MAX_PLAUSIBLE:
        out["band"] = "anomaly"
        cache[ckey] = out
        return out
    # GUARD 3 — "severe" additionally requires a materially large absolute loss,
    # so a small catchment can't reach the worst band on percentage alone.
    if pct <= RESALE_DECLINE_SEVERE and abs_loss >= POP_MIN_ABS_LOSS:
        band = "severe"
    elif pct <= RESALE_DECLINE_MILD:
        band = "mild"
    elif pct >= RESALE_GROWTH:
        band = "growth"
    else:
        band = "flat"
    out["band"] = band
    cache[ckey] = out
    return out

RESALE_GATE_CAP = 3.0   # max absolute points from the standalone resale gate

def resale_gate_adjustment(outlook: dict) -> float:
    """Standalone demographic resale adjustment, INDEPENDENT of floor area.

    Design choice (the handoff offered two): the population outlook is kept as
    its own gate rather than folded into exit-market breadth. Two reasons —
    ordering (the size model runs in analyze_properties, before coordinates and
    therefore before any outlook exists) and separability (breadth is measured
    from completed TRANSACTIONS, demographics from projections; keeping them
    apart means neither silently amplifies the other).

    It multiplies nothing: it is a small additive term on the final score, so it
    can never scale the legacy area penalty back into existence. Unknown,
    low-base and anomalous outlooks return exactly 0.0."""
    band = (outlook or {}).get("band")
    return {"severe": -RESALE_GATE_CAP, "mild": -RESALE_GATE_CAP / 2.0,
            "flat": 0.0, "growth": RESALE_GATE_CAP / 2.0}.get(band, 0.0)

def resale_multiplier(outlook: dict) -> float:
    """Scale factor applied to the floor-area exit-liquidity penalty.

    Unknown demographics give exactly 1.0 — never a discount, since absent data
    must not read as good news. The same neutrality applies to "low_base" (too
    few residents for a percentage to mean anything) and "anomaly" (implausible
    figure): a number we don't trust must not move the score in EITHER
    direction."""
    return RESALE_MULTIPLIERS.get((outlook or {}).get("band", "flat"), 1.0)

def _band_score(v: float, lo: float, hi: float, invert: bool = False) -> float:
    """Map v linearly onto 0–100 within [lo, hi], clamped. invert=True means
    lower raw values are better (station walk, building age)."""
    if math.isnan(v): return 50.0
    t = (v - lo) / (hi - lo) if hi != lo else 0.5
    t = min(1.0, max(0.0, t))
    return round((1.0 - t) * 100 if invert else t * 100, 1)

def analyze_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Score every listing out of 100 using the latest persisted system_weights.

    Sub-scores (each 0–100):
      yield_score       — gross yield against a 4–12% band (higher = better)
      station_proximity — walk minutes against a 1–20 min band (closer = better)
      asset_age         — building age against a 0–45 yr band (newer = better),
                          with an extra pre-1981 (kyū-taishin) penalty past 45 yrs
      price_efficiency  — listing ¥/m² divided by its prefecture's benchmark
                          ¥/m², scored on a 0.65×–1.35× band, inverted (priced
                          BELOW the local market per m² = better value)

    Composite = 100-point weighted blend, weights from get_latest_weights().
    The per-factor sub-scores are kept on the frame: the walk-forward loop later
    snapshots them so outcomes can be attributed to the right factors.
    """
    if df.empty:
        return df.assign(score=pd.Series(dtype=float))
    out = df.copy()
    # Net yield drives the yield factor: gross yield rewards high-fee buildings
    # for costs the owner actually pays. Computed for every row (fees estimated
    # from area when the listing's real figure isn't supplied); when net can't
    # be derived at all we fall back to the gross band so a missing figure is
    # never scored as zero.
    if "monthly_fees_yen" not in out.columns:
        out["monthly_fees_yen"] = None
    if "yield_basis" not in out.columns:
        out["yield_basis"] = YIELD_BASIS_DEFAULT
    _ny = out.apply(lambda r: compute_net_yield(
        r["price_yen"], r["gross_yield"], r["area_sqm"], r.get("ptype"),
        r.get("monthly_fees_yen"), r.get("yield_basis") or YIELD_BASIS_DEFAULT,
                            monthly_rent_yen=r.get("monthly_rent_yen"),
                            market_rent_yen=r.get("market_rent_yen")), axis=1)
    out["net_yield"] = [d["net_yield_pct"] for d in _ny]
    out["net_yield_invested"] = [d["net_yield_on_invested"] for d in _ny]
    out["fees_estimated"] = [d["fees_estimated"] for d in _ny]
    def _yield_score(r) -> float:
        # CALIBRATION. The band has to match the market actually being searched.
        # An earlier 2–9% net band was set for a national spread that includes
        # regional whole buildings at 8%+ net — but a central-Tokyo 区分マンション
        # nets 2.5–4%, so essentially every realistic listing scored near zero on
        # the heaviest-weighted factor and the whole scale collapsed.
        #   1% net  — marginal; you are buying for capital appreciation, not income
        #   6% net  — genuinely excellent for Japan
        # A typical Tokyo 1K therefore lands mid-scale, which is where a typical
        # deal belongs, and the factor can actually discriminate again.
        ny = safe_float(r["net_yield"])
        if not math.isnan(ny):
            return _band_score(ny, 1.0, 6.0)
        return _band_score(safe_float(r["gross_yield"]), 3.0, 9.0)
    out["yield_score"] = out.apply(_yield_score, axis=1)
    out["station_proximity"] = out["station_min"].apply(lambda v: _band_score(v, 1.0, 20.0, invert=True))
    # Pure age band. The old code also subtracted 10 points past 45 years as a
    # crude pre-1981 proxy; that's now a proper year-based 旧耐震 gate below, so
    # keeping it here would double-count the same risk.
    out["asset_age"] = out["building_age"].apply(lambda a: _band_score(a, 0.0, 45.0, invert=True))
    # ¥/m² relative to the local benchmark — now resolved at MUNICIPALITY
    # granularity: a listing carrying a city_code is scored against that ward's
    # real MLIT median (when refreshed), so a Shibuya unit is compared to
    # Shibuya, not a Tokyo-wide blend. Listings without a resolved city_code
    # (e.g. mock data) fall back to the prefecture benchmark inside
    # get_sqm_rate. Rates are cached per (city_code, prefecture) pair so we
    # never hit the database once per row.
    if "city_code" not in out.columns:
        out["city_code"] = None
    def _norm_cc(v) -> str | None:
        return v if (isinstance(v, str) and v.strip()) else None
    rate_cache: dict[tuple[str | None, str], float] = {}
    for cc, pf in {(_norm_cc(c), p) for c, p in zip(out["city_code"], out["prefecture"])}:
        rate_cache[(cc, pf)] = get_sqm_rate(cc, pref=pf)[0]
    def _ppsm_score(r) -> float:
        area = safe_float(r["area_sqm"])
        rate = rate_cache.get((_norm_cc(r.get("city_code")), r["prefecture"]))
        if not rate or math.isnan(area) or area <= 0:
            return 50.0   # unknown prefecture / missing area -> neutral midpoint
        return _band_score((r["price_yen"] / area) / rate, 0.65, 1.35, invert=True)
    out["price_efficiency"] = out.apply(_ppsm_score, axis=1)
    # 都市計画: zoning-class base (rebuild/income flexibility) blended with
    # designated FAR (容積率) capacity on a 100–500% band. Severely restricted
    # classes (市街化調整区域, 工業専用地域) crater the score; unknown zoning
    # is a neutral 50, never a penalty for missing information.
    def _dev_score(r) -> float:
        zoning = r.get("zoning") if isinstance(r.get("zoning"), str) else "unknown"
        base = ZONING_BASE_SCORES.get(zoning, 50.0)
        if zoning == "unknown":
            return 50.0
        far = far_or_nan(r.get("far_pct"))   # NULL/0/absent all collapse to NaN
        if math.isnan(far):
            return base                      # zoning known, FAR not listed -> zoning-only
        return round(0.6 * base + 0.4 * _band_score(far, 100.0, 500.0), 1)
    out["development_potential"] = out.apply(_dev_score, axis=1)

    w = get_latest_weights()
    # 旧耐震 gate: a categorical financing/resale risk, applied to the composite
    # outside the weighted factors (same treatment as the hazard gate). Applied
    # HERE rather than only in the analyzer so scan results and manually
    # analysed listings are scored on the same basis.
    _sc = out["building_age"].apply(lambda a: seismic_class(a)[0])
    out["seismic"] = _sc
    _pen = _sc.map({"old": get_old_seismic_penalty()}).fillna(0.0)
    # Legacy area classification remains available for compatibility and risk
    # wording. Only the selected size mode below determines its score effect.
    # Floor-area exit-liquidity gate: sub-20m² units are commonly ineligible for
    # investment loans, so the resale pool shrinks to cash buyers. Tenant demand
    # for small units is fine (single-person households are growing) — the risk
    # this prices is financing and RESALE, which ¥/m² alone can't see.
    _liq = out.apply(lambda r: area_liquidity(r["area_sqm"], r.get("ptype")), axis=1)
    out["area_tier"] = [d["tier"] for d in _liq]
    # SIZE TREATMENT. Exactly one of these applies — the modes are mutually
    # exclusive by construction, so the legacy deduction and the evidence-weighted
    # adjustment can never both land on the same score.
    mode = get_size_model_mode()
    out["size_model_mode"] = mode
    if mode == "legacy_penalty":
        _size_adj = -pd.Series([d["penalty"] for d in _liq], index=out.index)
        out["size_resilience"] = float("nan")
        out["size_evidence"] = float("nan")
        out["size_breakdown"] = None
    else:
        # Computed ONCE here, for both descriptive_only and evidence_weighted.
        # The full breakdown travels on the row as JSON so the panel, the
        # persisted value and the production adjustment are the same numbers
        # from the same inputs — not three independent recomputations.
        _res = out.apply(lambda r: size_resilience(
            r["area_sqm"], r.get("ptype"), city_code=r.get("city_code"),
            building_age=r.get("building_age"), structure=r.get("structure"),
            rent_evidence=_rent_evidence_from_row(r)), axis=1)
        out["size_resilience"] = [d["final"] if d["applicable"] else float("nan")
                                  for d in _res]
        out["size_evidence"] = [d.get("evidence", 0.0) for d in _res]
        out["size_breakdown"] = [json.dumps(d, ensure_ascii=False, default=str)
                                 for d in _res]
        _size_adj = (pd.Series(0.0, index=out.index) if mode == "descriptive_only"
                     else pd.Series([size_score_adjustment(d) for d in _res], index=out.index))
    # Round ONCE, then use that same rounded value everywhere. Previously the
    # score consumed the unrounded series while the persisted/displayed figure
    # was rounded, so the panel reported an adjustment that differed from the one
    # actually applied (by ~0.003). Small, but an audit row should reproduce the
    # score exactly, not approximately.
    _size_adj = _size_adj.round(2)
    out["size_adjustment"] = _size_adj
    out["score"] = (sum(out[f] * w[f] for f in FACTORS) - _pen + _size_adj
                    ).clip(lower=0.0, upper=100.0).round(1)
    return out.sort_values("score", ascending=False).reset_index(drop=True)

# ----------------------------------------------------------------------------
# The Autonomous Walk-Forward Feedback Loop
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Tab 1 — Top Deals
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Tab 2 — Market View
# ----------------------------------------------------------------------------
def render_ward_momentum_section() -> None:
    """📈 Ward momentum ranking: horizontal bar chart of annualized ¥/m² trend
    per ward, green up / red down, from stored MLIT quarterly medians. Purely
    informational (the trend is not in the score); hidden until data exists."""
    mom = get_all_ward_momentum()
    if mom.empty:
        return
    st.markdown(f"##### {tr('momentum_header')}")
    st.caption(tr("momentum_caption", n=len(mom)))
    chart_df = mom.assign(label=mom["ward"] + " (" + mom["pref"] + ")")
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("pct:Q", title=tr("momentum_axis")),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.condition(alt.datum.pct >= 0,
                                alt.value("#1a7f37"), alt.value("#cf222e")),
            tooltip=[alt.Tooltip("label:N", title=""),
                     alt.Tooltip("pct:Q", title="%/yr", format="+.1f"),
                     alt.Tooltip("quarters:Q", title="quarters"),
                     alt.Tooltip("samples:Q", title="MLIT samples")],
        )
        .properties(height=max(120, 26 * len(chart_df)))
    )
    st.altair_chart(chart, width="stretch")

def render_market_view() -> None:
    """Ward-level market data — momentum ranking and benchmark coverage.

    Everything here comes from real MLIT transaction records. The previous
    version of this tab filtered and charted randomly generated listings, which
    looked like analysis but told you nothing."""
    st.markdown(f"### {tr('market_header')}")
    st.caption(tr("market_intro"))

    render_ward_momentum_section()

    # Benchmark coverage: which wards actually have real data behind them, so
    # it is obvious when a score is resting on a prefecture average instead.
    st.markdown(f"##### {tr('coverage_header')}")
    with get_conn() as conn:
        cov = _read_sql(
            "SELECT city_code, avg_price_sqm, sample_count, updated_at "
            "FROM municipality_benchmarks WHERE avg_price_sqm > 0 "
            "ORDER BY avg_price_sqm DESC", conn)
    if cov.empty:
        st.info(tr("coverage_empty"))
        return
    cov["ward"] = cov["city_code"].map(municipality_label)
    cov["pref"] = cov["city_code"].map(lambda c: pref_name(_pref_key_from_city_code(c) or ""))
    disp = cov[["ward", "pref", "avg_price_sqm", "sample_count", "updated_at"]].rename(columns={
        "ward": tr("col_ward"), "pref": tr("col_prefecture"),
        "avg_price_sqm": tr("col_sqm_price"), "sample_count": tr("col_samples"),
        "updated_at": tr("col_updated")})
    st.dataframe(disp, width="stretch", hide_index=True, column_config={
        tr("col_sqm_price"): st.column_config.NumberColumn(tr("col_sqm_price"), format="¥%.0f"),
    })
    st.caption(tr("coverage_note", n=len(cov), total=sum(len(v) for v in MLIT_TARGET_CITIES.values())))

# ----------------------------------------------------------------------------
# Vision extraction — listing screenshots -> structured form values
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Geocoding — address text -> (lat, lon)
# ----------------------------------------------------------------------------
# Enables coordinate-keyed MLIT layers (hazard zones, land valuation, facility
# proximity) that can't be queried by city_code. GSI (国土地理院) is free, needs
# no key, and is tuned for Japanese addresses — a real advantage over generic
# geocoders here. Provider chosen by GEOCODER (secret/env): "gsi" | "mock".
# Default "mock" so the pipeline is exercisable offline; live geocoding is opt-in.
# GSI endpoint returns GeoJSON; geometry.coordinates is [lon, lat] (X,Y order).
GSI_GEOCODE_ENDPOINT = "https://msearch.gsi.go.jp/address-search/AddressSearch"
# Rough bounding box for the Japanese mainland the app covers, so a wildly
# off-result (bad parse, overseas hit) is rejected rather than silently stored.
JP_LAT_BOUNDS = (24.0, 46.0)
JP_LON_BOUNDS = (122.0, 146.0)

class GeocodeError(Exception):
    """Raised when an address can't be geocoded; callers degrade to no-coords
    (everything downstream stays neutral) rather than surfacing an error."""

def _geocode_gsi(address: str) -> tuple[float, float]:
    """Live GSI geocode. STUB STATUS: endpoint/response shape is current as of
    writing — verify against GSI's docs before relying on it. Returns (lat, lon)."""
    q = urllib.parse.urlencode({"q": address})
    req = urllib.request.Request(f"{GSI_GEOCODE_ENDPOINT}?{q}",
                                 headers={"User-Agent": "jp-re-screener/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    if not payload:
        raise GeocodeError("no geocode match")
    # GeoJSON feature list, best match first; coordinates are [lon, lat].
    coords = payload[0]["geometry"]["coordinates"]
    return float(coords[1]), float(coords[0])

def _geocode_mock(address: str) -> tuple[float, float]:
    """Offline demo: deterministic pseudo-coordinates inside the Tokyo area,
    derived from the address hash. NOT real positions — the UI flags this when
    GEOCODER is unset/mock so a user never mistakes demo coords for the truth."""
    h = int(hashlib.sha256(address.encode()).hexdigest()[:8], 16)
    lat = 35.60 + (h % 1000) / 10000.0          # ~35.60–35.70
    lon = 139.65 + ((h // 1000) % 1000) / 10000.0  # ~139.65–139.75
    return round(lat, 6), round(lon, 6)

def geocode_address(address: str) -> tuple[float, float] | None:
    """Address text -> (lat, lon), or None on any failure/empty input.

    Dispatches on GEOCODER (default 'mock'), caches per-address in session so a
    rerun never re-hits the service, validates the result is inside Japan, and
    NEVER raises — a miss returns None and everything downstream stays neutral.
    """
    if not address or not address.strip():
        return None
    cache = st.session_state.setdefault("_geocode_cache", {})
    key = address.strip()
    if key in cache:
        return cache[key]
    provider = (_get_secret("GEOCODER", "gsi") or "gsi").lower()
    fn = {"gsi": _geocode_gsi, "mock": _geocode_mock}.get(provider)
    result = None
    if fn is not None:
        try:
            lat, lon = fn(key)
            if JP_LAT_BOUNDS[0] <= lat <= JP_LAT_BOUNDS[1] and JP_LON_BOUNDS[0] <= lon <= JP_LON_BOUNDS[1]:
                result = (round(lat, 6), round(lon, 6))
        except Exception:
            result = None   # network/parse/empty -> degrade silently to None
    cache[key] = result
    return result

# ----------------------------------------------------------------------------
# Hazard gate — MLIT 不動産情報ライブラリ disaster-zone APIs (coordinate-keyed)
# ----------------------------------------------------------------------------
# Consumes the geocoded (lat, lon) to flag flood / landslide exposure. These are
# the same XYZ-tile GeoJSON APIs MLIT launched for disaster data in Nov 2025.
# Verified endpoint contract (XKT029 manual): params response_format=geojson,
# z (11–15), x, y; key in Ocp-Apim-Subscription-Key header; response may be gzip.
#
# DESIGN: this is a GATE with a warning banner, NOT a sixth weighted factor —
# a flood/landslide zone is a categorical risk (financing, insurance, resale),
# like the 市街化調整区域 zoning trap, so it applies a fixed score penalty and a
# prominent banner rather than competing smoothly in the gradient.
HAZARD_ENDPOINTS = {
    "flood":     "https://www.reinfolib.mlit.go.jp/ex-api/external/XKT026",  # 洪水浸水想定(最大規模)
    "landslide": "https://www.reinfolib.mlit.go.jp/ex-api/external/XKT029",  # 土砂災害警戒区域
    "surge":     "https://www.reinfolib.mlit.go.jp/ex-api/external/XKT027",  # 高潮浸水想定区域
    "tsunami":   "https://www.reinfolib.mlit.go.jp/ex-api/external/XKT028",  # 津波浸水想定
}
# Severity multipliers applied to the (tunable) hazard penalty, so a yellow
# sub-0.5m flood fringe is not treated like a 10m deep-red basin.
#   flood:     A31a_205 浸水深ランク 1..6 = <0.5m / 0.5-3 / 3-5 / 5-10 / 10-20 / >20m
#              (standard 水害ハザードマップ手引き thresholds — the map's colors)
#   landslide: A33_002 区域区分 1=警戒区域 (yellow), 2=特別警戒区域 (red)
#   surge/tsunami: presence-based at 1.0 for now — their rank schemas aren't
#              verified yet, and over-weighting them unverified would be worse
#              than neutral.
FLOOD_RANK_SEVERITY = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25, 5: 1.5, 6: 1.5}
LANDSLIDE_CLASS_SEVERITY = {1: 0.8, 2: 1.3}
HAZARD_TILE_Z = 15                  # max detail (manual allows 11–15)
HAZARD_PENALTY = 12.0               # DEFAULT points subtracted from the composite
                                    # (user-tunable in the KPI Weights tab; 18 was
                                    # too blunt — flood maps cover entire eastern-
                                    # Tokyo wards regardless of mapped depth)
DEV_BONUS = 4.0                     # DEFAULT points ADDED per development flag
                                    # (capped at 2 flags); tunable alongside it

def get_hazard_penalty() -> float:
    return float(st.session_state.get("hazard_penalty", HAZARD_PENALTY))

def get_dev_bonus() -> float:
    return float(st.session_state.get("dev_bonus", DEV_BONUS))
HAZARD_REQUEST_PAUSE_S = 0.25

def _latlon_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    """Slippy-map (XYZ) tile coordinate for a lat/lon at zoom z — the standard
    Web-Mercator formula the MLIT tile APIs use (links to GSI's tileCoordCheck)."""
    lat_r = math.radians(lat)
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y

def _point_in_ring(lon: float, lat: float, ring: list) -> bool:
    """Ray-casting point-in-polygon for one linear ring of [lon,lat] pairs."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and            (lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside

def _point_in_geometry(lon: float, lat: float, geom: dict) -> bool:
    """True if the point falls in a (Multi)Polygon geometry. Lines/points in the
    feed (some 土砂 data includes lines) can't contain a point, so they're skipped."""
    t = geom.get("type")
    coords = geom.get("coordinates") or []
    if t == "Polygon":
        return bool(coords) and _point_in_ring(lon, lat, coords[0])
    if t == "MultiPolygon":
        return any(poly and _point_in_ring(lon, lat, poly[0]) for poly in coords)
    return False

MLIT_MAX_RETRIES = 3        # attempts on HTTP 429 before giving up
MLIT_BACKOFF_BASE_S = 1.5   # 1.5s, 3s, 6s — respects Retry-After when present

# Statuses that mean "try again shortly", not "this request is wrong":
#   429 rate limited · 500/502/503/504 the provider is overloaded or restarting.
# Vision providers return 503 routinely under load, and a single un-retried
# attempt turns a momentary blip into a hard failure for the user.
TRANSIENT_HTTP = {429, 500, 502, 503, 504}

def vision_error_message(exc: Exception) -> str:
    """Turn a raw provider exception into something actionable.

    "HTTP Error 503: Service Unavailable" is accurate and useless: it looks like
    a broken app when it means the provider is briefly overloaded. Classify by
    status so the message says whether to wait, fix a key, or try another route."""
    code = getattr(exc, "code", None)
    if code in (500, 502, 503, 504):
        return tr("vision_provider_down", code=code)
    if code == 429:
        return tr("vision_rate_limited")
    if code in (401, 403):
        return tr("vision_bad_key", code=code)
    if code == 400:
        return tr("vision_bad_request")
    return tr("extract_failed", msg=_scrub_secrets(str(exc))[:160])

def _urlopen_with_backoff(req, timeout: int = 25, retries: int | None = None):
    """urlopen that retries TRANSIENT failures with exponential backoff.

    Retries 429 and 5xx; honours Retry-After when the server sends it. Anything
    else (401 bad key, 403 forbidden, 404 missing, 400 malformed) is raised
    immediately — those never get better by asking again, and retrying them just
    wastes the user's time and the provider's quota."""
    attempts = retries or MLIT_MAX_RETRIES
    delay = MLIT_BACKOFF_BASE_S
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code not in TRANSIENT_HTTP or attempt == attempts - 1:
                raise
            retry_after = safe_float((e.headers or {}).get("Retry-After"))
            time.sleep(min(retry_after if not math.isnan(retry_after) else delay, 30.0))
            delay *= 2
        except urllib.error.URLError:
            # DNS hiccup / connection reset: also worth one more try.
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise VisionError("exhausted retries")   # unreachable; keeps type checkers happy

def _fetch_hazard_tile(url: str, api_key: str, z: int, x: int, y: int) -> dict:
    """One XYZ GeoJSON hazard tile (isolated for offline monkeypatching). Handles
    the gzip-encoded response the MLIT manual documents, and backs off on 429."""
    q = urllib.parse.urlencode({"response_format": "geojson", "z": z, "x": x, "y": y})
    req = urllib.request.Request(f"{url}?{q}",
                                 headers={"Ocp-Apim-Subscription-Key": api_key,
                                          "User-Agent": "jp-re-screener/1.0"})
    with _urlopen_with_backoff(req, timeout=25) as resp:
        data = resp.read()
        if "gzip" in (resp.headers.get("Content-Encoding") or "").lower() \
                or data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
    return json.loads(data.decode("utf-8", errors="ignore"))

# Development-context layers (positive/informational — the mirror image of the
# hazard gate). Same MLIT tile APIs, same key, same caching discipline. NOTE the
# geometry difference: 高度利用地区 are POLYGONS (containment test), but 都市計画
# 道路 are LINES — a point is never "inside" a line, so roads use a proximity
# test (within DEV_ROAD_TOL_M of the planned corridor) instead.
DEV_ENDPOINTS = {
    "kodo":  "https://www.reinfolib.mlit.go.jp/ex-api/external/XKT024",  # 高度利用地区
    "road":  "https://www.reinfolib.mlit.go.jp/ex-api/external/XKT030",  # 都市計画道路
    "chiku": "https://www.reinfolib.mlit.go.jp/ex-api/external/XKT023",  # 地区計画
}
# HONEST LIMITATION, worth stating plainly: many of Japan's highest-profile
# redevelopments (Shibuya Scramble Square, Osaka's "Brillia Tower Dojima" /
# Four Seasons complex — confirmed via Tokyo Tatemono's own press material to
# be a 特定街区 city-planning decision) are designated via 都市再生特別地区
# (Urban Renaissance Special District) or 特定街区 (Specified Block), NOT
# 高度利用地区 or 地区計画. MLIT's reinfolib API does not expose either of
# those two designation types at all (checked the full XKT catalog). So a
# major tower can legitimately show NO development flag here even though it
# is, in the real world, exactly the kind of project this feature is meant to
# surface — the gap is in what the data source publishes, not in this code.
# The banner's "verify at the ward's 都市計画課" line exists for precisely
# this case.
DEV_ROAD_TOL_M = 30.0   # a planned road within ~30m materially affects the parcel
DEV_NEAR_M = 300.0      # "nearby" tier: a 高度利用地区 boundary within ~300m —
                        # the neighbor-block signal (worth half a flag)
DEV_ACTIVITY_MIN = 3    # absolute fallback threshold for the "active area" tier:
                        # this many development features in the listing's ~1km
                        # tile. Becomes RELATIVE (2x the observed median) once
                        # >= 10 tiles of real data have accumulated — the
                        # baseline builds itself from the user's own analyses.

def _point_near_line(lon: float, lat: float, coords: list, tol_m: float) -> bool:
    """True if (lon,lat) lies within tol_m of any segment of a [lon,lat] line.
    Uses a local equirectangular approximation (fine at parcel scale)."""
    mx = 111_320.0 * math.cos(math.radians(lat))   # metres per degree lon here
    my = 111_320.0                                  # metres per degree lat
    px, py = lon * mx, lat * my
    for i in range(len(coords) - 1):
        ax, ay = coords[i][0] * mx, coords[i][1] * my
        bx, by = coords[i + 1][0] * mx, coords[i + 1][1] * my
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        qx, qy = ax + t * dx, ay + t * dy
        if math.hypot(px - qx, py - qy) <= tol_m:
            return True
    return False

def _polygon_boundary_near(lon: float, lat: float, geom: dict, tol_m: float) -> bool:
    """True if (lon,lat) lies within tol_m of a (Multi)Polygon's exterior ring —
    the 'zone starts a block away' signal. Containment is tested separately;
    this is for points OUTSIDE the zone but close to its edge."""
    t = geom.get("type")
    coords = geom.get("coordinates") or []
    if t == "Polygon" and coords:
        return _point_near_line(lon, lat, coords[0], tol_m)
    if t == "MultiPolygon":
        return any(poly and _point_near_line(lon, lat, poly[0], tol_m) for poly in coords)
    return False

def _point_hits_dev_geometry(lon: float, lat: float, geom: dict) -> bool:
    """Containment for (Multi)Polygon zones; proximity for (Multi)LineString roads."""
    t = geom.get("type")
    coords = geom.get("coordinates") or []
    if t in ("Polygon", "MultiPolygon"):
        return _point_in_geometry(lon, lat, geom)
    if t == "LineString":
        return _point_near_line(lon, lat, coords, DEV_ROAD_TOL_M)
    if t == "MultiLineString":
        return any(_point_near_line(lon, lat, line, DEV_ROAD_TOL_M) for line in coords)
    return False

def check_development(lat: float | None, lon: float | None) -> dict:
    """{dev_key: bool} for development-context layers at (lat, lon). Same
    contract as check_hazards: needs MLIT_API_KEY + coords, per-hazard error
    isolation (failed key absent = unknown, never a fabricated answer), caches
    only fully-determined results, NEVER raises. Display-only — these flags do
    not enter the score."""
    out: dict[str, bool] = {}
    if lat is None or lon is None:
        return out
    api_key = _get_secret("MLIT_API_KEY")
    if not api_key:
        return out
    cache = st.session_state.setdefault("_dev_cache", {})
    ckey = f"{round(lat,5)},{round(lon,5)}"
    if ckey in cache:
        return cache[ckey]
    x, y = _latlon_to_tile(lat, lon, HAZARD_TILE_Z)
    complete = True
    counts: dict[str, int] = {}
    for key, url in DEV_ENDPOINTS.items():
        try:
            payload = _fetch_hazard_tile(url, api_key, HAZARD_TILE_Z, x, y)
            feats = payload.get("features", []) if isinstance(payload, dict) else []
            counts[key] = len(feats)
            out[key] = any(_point_hits_dev_geometry(lon, lat, f.get("geometry", {})) for f in feats)
            # Neighbor-block tier for zone polygons: NOT inside, but the zone
            # boundary is within DEV_NEAR_M. Same tile data — zero extra calls.
            if key in ("kodo", "chiku") and not out[key]:
                out[f"{key}_near"] = any(
                    _polygon_boundary_near(lon, lat, f.get("geometry", {}), DEV_NEAR_M)
                    for f in feats)
        except Exception:
            complete = False
        time.sleep(HAZARD_REQUEST_PAUSE_S)
    if complete:
        # "Active development area" tier: total development features in the
        # listing's ~1km tile vs the (self-accumulating) baseline.
        total = sum(counts.values())
        out["activity"] = total >= _dev_activity_threshold()
        _store_dev_tile_stats(x, y, counts.get("kodo", 0), counts.get("road", 0))
        cache[ckey] = out
    return out

def _dev_activity_threshold() -> int:
    """Threshold for the 'active area' flag. Starts at DEV_ACTIVITY_MIN; once
    >= 10 tiles of real observations exist, becomes 2x the observed median
    (never below the absolute minimum) — i.e. 'well above average for the areas
    you actually look at', which is the honest version of 'above average'
    without a metro-wide tile sweep we can't do."""
    def _load():
        try:
            with get_conn() as conn:
                rows = conn.execute(
                    "SELECT kodo_count + road_count AS t FROM dev_tile_stats").fetchall()
            totals = sorted(int(r["t"]) for r in rows)
            if len(totals) < 10:
                return DEV_ACTIVITY_MIN
            median = totals[len(totals) // 2]
            return max(DEV_ACTIVITY_MIN, 2 * median)
        except Exception:
            return DEV_ACTIVITY_MIN
    return _memo("dev_threshold", _load)

def _store_dev_tile_stats(x: int, y: int, kodo_count: int, road_count: int) -> None:
    """Accumulate per-tile development counts — the raw material for the
    relative activity baseline. Best-effort; never raises."""
    try:
        with get_conn() as conn:
            conn.execute(
                _upsert_sql("dev_tile_stats",
                            ["z", "x", "y", "kodo_count", "road_count", "checked_at"],
                            conflict=["z", "x", "y"]),
                (HAZARD_TILE_Z, int(x), int(y), int(kodo_count), int(road_count),
                 datetime.now().isoformat(timespec="seconds")))
        _memo_invalidate("dev_threshold")
    except Exception:
        pass

def check_hazards(lat: float | None, lon: float | None) -> dict:
    """Return {hazard_key: bool} for each configured zone at (lat, lon).

    Requires MLIT_API_KEY and coordinates. Tile-based APIs return HTTP 200 with
    an EMPTY feature array when a tile has no data (per the manual's Q.7), so
    'no features' correctly means 'not in a zone'. Each hazard is independently
    guarded; any failure yields a missing key (treated as unknown, never a false
    'safe'). Cached per (lat,lon) in session. NEVER raises.
    """
    out: dict[str, dict] = {}
    if lat is None or lon is None:
        return out
    api_key = _get_secret("MLIT_API_KEY")
    if not api_key:
        return out
    cache = st.session_state.setdefault("_hazard_cache", {})
    ckey = f"{round(lat,5)},{round(lon,5)}"
    if ckey in cache:
        return cache[ckey]
    x, y = _latlon_to_tile(lat, lon, HAZARD_TILE_Z)
    complete = True
    for hz, url in HAZARD_ENDPOINTS.items():
        try:
            payload = _fetch_hazard_tile(url, api_key, HAZARD_TILE_Z, x, y)
            feats = payload.get("features", []) if isinstance(payload, dict) else []
            # The point is "in a zone" only if it actually falls inside one of
            # the returned polygons — a tile can contain zones the listing isn't in.
            hits = [f for f in feats
                    if _point_in_geometry(lon, lat, f.get("geometry", {}))]
            out[hz] = _grade_hazard(hz, hits)
        except Exception:
            complete = False   # transient failure -> leave key absent ('unknown')
        time.sleep(HAZARD_REQUEST_PAUSE_S)
    # Only cache a fully-determined result. A transient network failure must NOT
    # pin "unknown" for the whole session — a later rerun should retry.
    if complete:
        cache[ckey] = out
    return out

def _grade_hazard(hz: str, hits: list[dict]) -> dict:
    """Grade the WORST containing zone: {'hit', 'sev', 'detail'}.

    Flood zones carry a depth rank (A31a_205); landslide zones a class
    (A33_002, 特別警戒=2 is the red one). Missing/unparseable attributes fall
    back to severity 1.0 — grading refines the penalty, its absence must never
    soften a known hit below the baseline."""
    if not hits:
        return {"hit": False, "sev": 0.0, "detail": None}
    sev, detail = 1.0, None
    if hz == "flood":
        ranks = []
        for f in hits:
            try:
                ranks.append(int((f.get("properties") or {}).get("A31a_205")))
            except (TypeError, ValueError):
                pass
        if ranks:
            worst = max(ranks)
            sev = FLOOD_RANK_SEVERITY.get(worst, 1.0)
            detail = f"rank{min(worst, 6)}"
    elif hz == "landslide":
        classes = []
        for f in hits:
            try:
                classes.append(int((f.get("properties") or {}).get("A33_002")))
            except (TypeError, ValueError):
                pass
        if classes:
            worst = max(classes)
            sev = LANDSLIDE_CLASS_SEVERITY.get(worst, 1.0)
            detail = f"class{min(worst, 2)}"
    return {"hit": True, "sev": sev, "detail": detail}

class PortalBlockedError(Exception):
    """The portal refused an automated fetch (HTTP 403/401/429) — an access
    policy decision, not a recoverable error. The UI routes the user to the
    screenshot path, which involves no server request and so is never blocked."""

class VisionError(Exception):
    """Raised when a vision provider can't produce a usable extraction."""

# The extraction contract sent to every provider. JSON-only output keeps the
# downstream parsing trivial; unknown fields are nulled, never guessed.
_VISION_PROMPT = """You read Japanese real-estate listing screenshots (RENOSY, 健美家, 楽待, mobile or desktop).
Extract the listing details and reply with ONLY a JSON object — no prose, no markdown fences — with exactly these keys (use null when a value is not visible):
{
  "price_man": <asking price in 万円 as a number, e.g. ¥19,000,000 -> 1900>,
  "monthly_rent_yen": <monthly rent in yen, e.g. ¥74,000 -> 74000>,
  "gross_yield_pct": <表面利回り as a number, e.g. 4.67% -> 4.67>,
  "area_sqm": <private-use / floor area 専有面積 in m²>,
  "station_walk_min": <the SMALLEST 徒歩 minutes among all listed stations; ignore バス minutes>,
  "building_age_years": <age in years; derive from completion date if needed>,
  "layout": <間取り string exactly as shown, e.g. "1K">,
  "address_jp": <the FULL address as printed, e.g. "東京都文京区本駒込4-40-6" (used for geocoding)>,
  "prefecture_jp": <prefecture from the address, e.g. "東京都">,
  "city_jp": <municipality (市区町村) from the address. For a designated city (政令指定都市) ALWAYS include the parent city, e.g. "横浜市西区", "大阪市北区", "相模原市南区" — NEVER the bare ward ("西区"). For an ordinary ward/city use it as-is, e.g. "渋谷区", "習志野市".>,
  "portal_guess": <"Renosy" | "Kenbiya" | "Rakumachi" | null>,

  "building_name": <物件名/建物名/マンション名 exactly as printed, or null>,
  "unit_number": <部屋番号 e.g. "603", or null if not shown>,
  "unit_floor": <所在階 as a number, e.g. "13階/14階建" -> 13>,
  "total_floors": <total floors of the building, e.g. "13階/14階建" -> 14>,
  "balcony_sqm": <バルコニー面積 in m², or null>,
  "structure": <建物構造 as one of "RC" | "SRC" | "S" | "Wood" | null. 鉄筋コンクリート->RC, 鉄骨鉄筋コンクリート->SRC, 鉄骨->S, 木造->Wood>,
  "total_units": <総戸数 as a number, or null>,
  "land_rights": <土地権利 e.g. "所有権" | "借地権", or null>,
  "zoning_jp": <用途地域 exactly as printed, or null>,
  "far_pct": <容積率 as a number, e.g. "400%" -> 400>,

  "monthly_fees_yen": <管理費 in yen per month, or null>,
  "repair_reserve_yen": <修繕積立金 in yen per month, or null>,
  "market_rent_yen": <想定賃料/想定家賃 (an ESTIMATE of achievable rent) in yen per month, or null>,
  "yield_basis": <"current" if the yield is labelled 現況利回り; "full" if 満室時/想定/表面利回り; else null>,
  "occupancy": <"rented" if 賃貸中/入居中, "vacant" if 空室, else null>,

  "source_document_date": <date printed on the document in YYYY-MM-DD, or null>,
  "company_roles": <array of [label, company_name] pairs for companies whose ROLE
     is printed next to them, e.g. [["管理会社","株式会社エステム管理サービス"],
     ["仲介","株式会社アルファ不動産"]]. Include the label EXACTLY as printed.>,
  "extracted_entities": <array of {"name": <company>, "label": "", "role": "unknown"}
     for any company whose role is NOT printed>
}

CRITICAL RULES
- Use null for anything not visible. Never guess, never carry a value over from
  another property shown on the same page.
- monthly_rent_yen is the CURRENT CONTRACTUAL rent (現行賃料/賃料/家賃). It is a
  DIFFERENT field from market_rent_yen (想定賃料), which is only an estimate.
  Never put an estimate in monthly_rent_yen.
- Do NOT infer an agency from an unlabelled company name. A company printed
  without a role goes in extracted_entities with role "unknown" — a management
  company (管理会社) is not the brokerage and must never be reported as one.
- If several properties appear (comparison tables, recommendation strips),
  extract ONLY the main subject property."""

def _vision_openai(image_b64: str, mime: str) -> dict:
    """OpenAI vision call (REST, stdlib only). Needs OPENAI_API_KEY; model
    overridable via VISION_MODEL_OPENAI. STUB STATUS: verify endpoint/model
    against OpenAI's current docs before relying on it."""
    key = _get_secret("OPENAI_API_KEY")
    if not key:
        raise VisionError("OPENAI_API_KEY not set")
    model = _get_secret("VISION_MODEL_OPENAI", "gpt-4o")
    body = json.dumps({
        "model": model,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ]}],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with _urlopen_with_backoff(req, timeout=60, retries=4) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return _json_from_text(payload["choices"][0]["message"]["content"])

def _vision_gemini(image_b64: str, mime: str) -> dict:
    """Gemini vision call (REST, stdlib only). Needs GEMINI_API_KEY; model
    overridable via VISION_MODEL_GEMINI. STUB STATUS: verify endpoint/model
    against Google's current docs."""
    key = _get_secret("GEMINI_API_KEY")
    if not key:
        raise VisionError("GEMINI_API_KEY not set")
    model = _get_secret("VISION_MODEL_GEMINI", "gemini-2.5-flash")
    # Key goes in the x-goog-api-key HEADER, never the URL query string: a
    # urllib URLError stringifies the requested URL, so a query-string key
    # could leak into logs or the st.error path. Google supports the header
    # for exactly this reason.
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    body = json.dumps({"contents": [{"parts": [
        {"text": _VISION_PROMPT},
        {"inline_data": {"mime_type": mime, "data": image_b64}},
    ]}]}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json",
                                          "x-goog-api-key": key})
    with _urlopen_with_backoff(req, timeout=60, retries=4) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return _json_from_text(payload["candidates"][0]["content"]["parts"][0]["text"])

def _vision_anthropic(image_b64: str, mime: str) -> dict:
    """Anthropic vision call (REST, stdlib only). Needs ANTHROPIC_API_KEY;
    model overridable via VISION_MODEL_ANTHROPIC."""
    key = _get_secret("ANTHROPIC_API_KEY")
    if not key:
        raise VisionError("ANTHROPIC_API_KEY not set")
    # NOTE: a prior external review flagged this model string as a
    # "hallucination" and suggested claude-3-5-sonnet-20241022 — that advice
    # was itself stale (reviewer knowledge cutoff). Claude 4.x is the current
    # naming line; default below is current as of mid-2026 and overridable via
    # the VISION_MODEL_ANTHROPIC secret precisely because model names drift.
    model = _get_secret("VISION_MODEL_ANTHROPIC", "claude-sonnet-4-6")
    body = json.dumps({
        "model": model, "max_tokens": 500,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": image_b64}},
            {"type": "text", "text": _VISION_PROMPT},
        ]}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"})
    with _urlopen_with_backoff(req, timeout=60, retries=4) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return _json_from_text("".join(b.get("text", "") for b in payload.get("content", [])))

def _vision_mock(image_b64: str, mime: str) -> dict:
    """Offline demo provider: CANNED values (a Tokyo Shinbashi 1K) so the whole
    pipeline is exercisable without any API key. The UI shows an explicit
    warning that these are not from the user's screenshot."""
    return {"price_man": 2910, "monthly_rent_yen": 92000, "gross_yield_pct": 3.79,
            "area_sqm": 18.75, "station_walk_min": 5, "building_age_years": 22,
            "layout": "1K", "prefecture_jp": "東京都", "city_jp": "港区",
            "address_jp": "東京都港区新橋5-30-5", "portal_guess": "Renosy"}

def _json_from_text(text: str) -> dict:
    """Parse a JSON object out of an LLM reply, tolerating ```json fences."""
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise VisionError("no JSON object in model reply")
    try:
        out = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as e:
        raise VisionError(f"bad JSON from model: {e}") from e
    if not isinstance(out, dict):
        raise VisionError("model reply was not a JSON object")
    return out

def extract_listing_from_image(image_bytes: bytes, mime: str) -> dict:
    """Dispatch to the configured vision provider (VISION_PROVIDER secret/env:
    'openai' | 'gemini' | 'anthropic' | 'mock'; default 'mock')."""
    provider = (_get_secret("VISION_PROVIDER", "mock") or "mock").lower()
    image_b64 = base64.b64encode(image_bytes).decode()
    fn = {"openai": _vision_openai, "gemini": _vision_gemini,
          "anthropic": _vision_anthropic, "mock": _vision_mock}.get(provider)
    if fn is None:
        raise VisionError(f"unknown VISION_PROVIDER '{provider}'")
    return fn(image_b64, mime)

def parse_extension_payload(text: str) -> dict:
    """Map the browser extension's DOM-scraped JSON into the raw-extraction dict
    that normalize_extraction consumes. The extension reads the page the USER is
    already viewing while logged in and the user manually triggers it — no
    server request, no bot — but the payload is still untrusted text, so it goes
    through the identical trust boundary as vision extraction (every field
    clamped/typed downstream). Raises VisionError on malformed input.

    Expected JSON (all optional; missing fields just don't pre-fill):
      {"price_man": 6290, "area_sqm": 25.3, "station_walk_min": 5,
       "building_age_years": 12, "layout": "1K", "gross_yield_pct": 4.0,
       "address_jp": "東京都文京区本駒込4-40-6", "prefecture_jp": "東京都",
       "city_jp": "文京区", "zoning_jp": "商業地域", "far_pct": 400,
       "portal_guess": "Kenbiya"}
    """
    obj = _json_from_text(text)            # reuses the fenced-JSON extractor + guards
    if not isinstance(obj, dict):
        raise VisionError("extension payload is not a JSON object")
    # Pass through only known keys; ignore anything extra the page added.
    known = {"price_man", "area_sqm", "station_walk_min", "building_age_years",
             "layout", "gross_yield_pct", "yield_basis", "monthly_fees_yen",
             "monthly_rent_yen", "address_jp",
             "prefecture_jp", "city_jp", "zoning_jp", "far_pct", "portal_guess"}
    return {k: obj[k] for k in known if k in obj}

def normalize_extraction(raw: dict) -> tuple[dict, list[tuple[str, dict]]]:
    """Turn an LLM extraction into validated man_* session-state updates.

    THE TRUST BOUNDARY: everything from the model is treated as untrusted —
    coerced through safe_float, clamped to each widget's bounds, and dropped
    (with a localized warning) when unmappable, so a hallucinated value can
    never crash a widget or smuggle in an out-of-range number. Returns
    (updates, warnings) where warnings are (tr_key, kwargs) pairs rendered in
    the active language at display time."""
    updates: dict = {}
    warns: list[tuple[str, dict]] = []

    price_man = safe_float(raw.get("price_man"))
    if not math.isnan(price_man) and price_man > 0:
        updates["man_price"] = int(min(price_man, 1_000_000))   # ¥100億 cap

    area = safe_float(raw.get("area_sqm"))
    if not math.isnan(area) and area > 0:
        updates["man_area"] = float(min(area, 5000.0))

    station = safe_float(raw.get("station_walk_min"))
    if not math.isnan(station) and station > 0:
        updates["man_station"] = int(min(max(station, 1), 60))
        # A real extracted value must clear the unknown flag, or the form keeps
        # claiming the field is unknown while displaying the number it found.
        updates["man_station_unknown"] = False

    age = safe_float(raw.get("building_age_years"))
    if not math.isnan(age) and age >= 0:
        updates["man_age"] = int(min(age, 80))
        updates["man_age_unknown"] = False

    gy = safe_float(raw.get("gross_yield_pct"))
    rent = safe_float(raw.get("monthly_rent_yen"))
    if not math.isnan(gy) and 0 < gy <= 30:
        updates["man_yield"] = float(round(gy, 2))
    elif not math.isnan(rent) and rent > 0 and "man_price" in updates:
        calc = rent * 12.0 / (updates["man_price"] * 10_000.0) * 100.0
        if 0 < calc <= 30:
            updates["man_yield"] = float(round(calc, 2))
            warns.append(("warn_yield_computed", {"gy": calc, "rent": rent}))

    layout = str(raw.get("layout") or "").upper()
    if layout:
        for needle, ptype in LAYOUT_TO_PTYPE:
            if needle.upper() in layout:
                updates["man_ptype"] = ptype
                break
        else:
            warns.append(("warn_layout_unknown", {"name": raw.get("layout")}))

    yb = str(raw.get("yield_basis") or "").strip().lower()
    if yb in YIELD_BASES:
        updates["man_yield_basis"] = yb
    fees_in = safe_float(raw.get("monthly_fees_yen"))
    if not math.isnan(fees_in) and fees_in > 0:
        updates["man_fees"] = int(min(fees_in, 500_000))

    zoning_jp = str(raw.get("zoning_jp") or "").strip()
    if zoning_jp:
        ja_names = ZONING_NAMES["ja"]
        for key in sorted(ja_names, key=lambda k: -len(ja_names[k])):
            canon = ja_names[key].replace(" ⚠", "")
            if key != "unknown" and canon and canon in zoning_jp:
                updates["man_zoning"] = key
                break
    far = safe_float(raw.get("far_pct"))
    if not math.isnan(far) and far > 0:
        updates["man_far"] = int(min(far, 1300))

    pref_jp = str(raw.get("prefecture_jp") or "")
    if pref_jp:
        for needle, key in PREF_JP_MAP.items():
            if needle in pref_jp:
                updates["man_pref"] = key
                break
        else:
            warns.append(("warn_pref_unsupported", {"name": pref_jp}))

    # Resolve the hyper-local 市区町村コード from the extracted municipality. We
    # prefer the prefecture we just normalized (an internal key the resolution
    # table is keyed by); if that wasn't recognized we still try the raw
    # Japanese prefecture string. A miss simply leaves city_code unset, and the
    # scoring engine falls back to the prefecture benchmark — never an error.
    # NOTE: for designated-city wards (政令市), the resolver needs the FULL
    # compound ("横浜市西区", "相模原市南区") — a bare "南区" is ambiguous across
    # cities and correctly resolves to None (→ prefecture fallback). The vision/
    # URL prompt therefore asks for the full municipality incl. the parent city.
    city_jp = str(raw.get("city_jp") or "")
    if city_jp:
        pref_for_city = updates.get("man_pref") or pref_jp
        code = resolve_municipality_code(pref_for_city, city_jp)
        if code:
            updates["man_city_code"] = code

    addr = str(raw.get("address_jp") or "").strip()
    if addr:
        # Defense in depth (any ingestion path): if the address's own prefecture
        # contradicts the extracted prefecture_jp, it's almost certainly a
        # footer/company address rather than the listing's — drop it and warn,
        # rather than geocode the wrong place. Mirrors the extension's cross-check.
        addr_pref = None
        for needle, key in PREF_JP_MAP.items():
            full = needle + ("都" if needle == "東京" else "府" if needle in ("大阪", "京都") else "県")
            if addr.startswith(needle) or addr.startswith(full):
                addr_pref = key
                break
        want_pref = updates.get("man_pref")
        if want_pref and addr_pref and addr_pref != want_pref:
            warns.append(("warn_address_mismatch", {"addr_pref": pref_name(addr_pref),
                                                    "listing_pref": pref_name(want_pref)}))
        else:
            updates["man_address"] = addr[:120]   # bound length; it's free-text input

    portal = raw.get("portal_guess")
    if portal in MANUAL_PORTALS:
        updates["man_portal"] = portal
    # WS3/WS5 identity + attribution + rents. Text passes through as-is; numeric
    # fields are clamped. Market rent is carried SEPARATELY and never written to
    # the contractual-rent key.
    for src_key, dst_key in (
        ("building_name", "man_building_name"), ("unit_number", "man_unit_number"),
        ("unit_floor", "man_unit_floor"),
        ("source_document_name", "man_source_document_name"),
        ("source_document_date", "man_source_document_date"),
        ("ingestion_channel", "man_ingestion_channel"),
        ("source_type", "man_source_type"), ("agency_name", "man_agency_name"),
        ("agent_name", "man_agent_name"), ("agency_phone", "man_agency_phone"),
        ("agency_address", "man_agency_address"), ("agency_role", "man_agency_role"),
        ("seller_name", "man_seller_name"),
        ("management_company", "man_management_company"),
        ("rent_guarantee_company", "man_rent_guarantee_company"),
        ("occupancy", "man_occupancy"), ("structure", "man_structure"),
        # Pasted-listing additions.
        ("listing_title", "man_listing_title"), ("layout", "man_layout"),
        ("orientation", "man_orientation"), ("land_right", "man_land_right"),
        ("management_type", "man_management_type"),
        ("management_number", "man_management_number"),
        ("handover", "man_handover"), ("registered_at", "man_registered_at"),
        ("updated_at", "man_listing_updated_at"),
        ("next_update_at", "man_next_update_at"),
        ("construction_date", "man_construction_date"),
    ):
        val = raw.get(src_key)
        if val is not None and str(val).strip():
            updates[dst_key] = str(val).strip()[:120]
    for src_key, dst_key in (("monthly_rent_yen", "man_monthly_rent"),
                             ("market_rent_yen", "man_market_rent")):
        v = safe_float(raw.get(src_key))
        if not math.isnan(v) and v > 0:
            updates[dst_key] = int(min(v, 10_000_000))
    if raw.get("extracted_entities"):
        updates["man_extracted_entities"] = raw["extracted_entities"]

    return updates, warns

def recover_document_fields(raw: dict, page_text: str) -> dict:
    """Deterministic recovery for Japanese sales sheets (販売図面).

    These are usually Excel or DTP exports: a label sits in one cell and its
    value in another, so the extracted text separates them by long runs of
    spaces or a newline. The general-purpose patterns allow only a few
    characters between label and value, which is why a page containing every
    field in plain text yielded almost nothing. Here the gap is widened
    deliberately and each label is anchored, so a wide gap cannot let one
    label capture the next field's number.

    Existing values always win: this only fills what is still missing, so a
    vision result or a user correction is never overwritten.
    """
    out = dict(raw or {})
    text = unicodedata.normalize("NFKC", page_text or "")
    flat = re.sub(r"[ \t]+", " ", text)

    def put(key, value):
        if value is not None and out.get(key) in (None, "", [], {}):
            out[key] = value

    def money(label):
        m = re.search(label + r"[^0-9]{0,30}([0-9,]{3,})\s*円", flat, re.S)
        return int(m.group(1).replace(",", "")) if m else None

    def number(pattern, cast=float):
        m = re.search(pattern, flat, re.S)
        return cast(m.group(1).replace(",", "")) if m else None

    put("price_man", number(r"価格[^0-9]{0,30}([0-9,]+)\s*万円", float))
    put("area_sqm", number(r"専有面積[^0-9]{0,20}([0-9.]+)\s*m?[2²㎡]", float))
    put("balcony_sqm", number(r"バルコニー面積[^0-9]{0,20}([0-9.]+)\s*m?[2²㎡]", float))
    # Contractual vs market rent stay strictly separated. 相場賃料 ("prevailing
    # market rent") is an ESTIMATE and must never reach monthly_rent_yen.
    put("monthly_rent_yen", money(r"(?:現行賃料|契約賃料|現況賃料|家賃)"))
    put("market_rent_yen", money(r"(?:相場賃料|市場賃料|参考賃料|想定賃料)"))
    mgmt = money(r"管理費(?:\s*\(月額\)|\s*月額)?")
    reserve = money(r"修繕積立金(?:\s*\(月額\)|\s*月額)?")
    if mgmt:
        put("management_fee_yen", mgmt)
    if reserve:
        put("repair_reserve_yen", reserve)
    if mgmt or reserve:
        put("monthly_fees_yen", (mgmt or 0) + (reserve or 0))

    # Several stations are normal. Scoring takes one number, so use the shortest
    # walk, but keep every option so the choice is visible rather than implied.
    walks = [int(v) for v in re.findall(r"駅\s*徒歩\s*([0-9]{1,2})\s*分", flat)]
    if walks:
        # OVERRIDE, not put(): the generic parser takes the FIRST 徒歩 figure it
        # meets, which on a multi-station sheet is whichever line happens to be
        # printed first (here 大国町駅 6分 above 今宮駅 5分). The shortest walk is
        # the correct single value for scoring, so the specific answer wins over
        # the generic one. Both options are kept so the choice stays visible.
        out["station_walk_min"] = min(walks)
        out["station_walk_options"] = walks

    # 築年数 on these sheets is a construction DATE, not a count of years.
    built = re.search(r"(?:築年数|築年月|竣工|完成)[^0-9]{0,20}([12][0-9]{3})\s*年\s*([0-9]{1,2})\s*月",
                      flat, re.S)
    if built:
        year, month = map(int, built.groups())
        put("construction_date", f"{year:04d}-{month:02d}")
        today = date.today()
        put("building_age_years", today.year - year - (today.month < month))

    addr = re.search(r"((?:東京都|大阪府|京都府|北海道|.{2,3}県).{2,50}?[0-9]+(?:丁目[0-9-]+|[-−][0-9-]+))",
                     flat)
    if addr:
        put("address_jp", addr.group(1).strip())

    # Building name: prefer an explicit label; otherwise take the longest
    # katakana/kanji run from the first two lines, which is where these sheets
    # put the property title. Deliberately NOT keyed to one developer's brand —
    # a pattern that only matches エステムコート would silently fail on every
    # other sheet while appearing to work here.
    bname = re.search(r"(?:物件名|建物名|マンション名)\s*[:：]?\s*([^\n]{3,80})", page_text or "")
    if bname:
        put("building_name", re.sub(r"\s+", " ", bname.group(1)).strip(" ■◆●・★☆"))
    else:
        # Marketing copy sits in the same header band as the property name and
        # is usually LONGER, so "take the longest line" picks the slogan. These
        # sheets decorate copy with ☆★ and 、。 — the name itself does not. The
        # name also often wraps across two lines
        # (エステムコート難波 / サウスプレイスⅢラ・パーク), so adjacent
        # fragments in the same column are joined.
        # Names are read from the RAW text, not the NFKC-normalised copy. NFKC
        # is correct for numbers (full-width １２５ -> 125) but destructive for a
        # proper name: it rewrites エステムコート…Ⅲ as …III, so the stored name no
        # longer matches the document or the portal listing.
        cands = []
        for ln in (page_text or "").splitlines()[:8]:
            for piece in re.split(r"[ \t　]{2,}", ln):
                piece = piece.strip(" ■◆●・")
                if not piece or len(piece) < 4:
                    continue
                if re.search(r"[☆★、。]|価格|万円|所在地|徒歩|[0-9]{3,}", piece):
                    continue          # marketing copy or a labelled field
                if re.search(r"[ァ-ヶー]", piece):   # katakana => a product name
                    cands.append(piece)
        if cands:
            name = cands[0]
            if len(cands) > 1 and len(cands[0]) + len(cands[1]) <= 60:
                name = cands[0] + cands[1]        # wrapped across two lines
            put("building_name", name)

    if "RC造" in flat:
        put("structure", "RC")
    elif "SRC造" in flat:
        put("structure", "SRC")
    elif "鉄骨造" in flat:
        put("structure", "S")
    if "賃貸中" in flat or "入居中" in flat:
        put("occupancy", "rented")
    elif "空室" in flat:
        put("occupancy", "vacant")

    manager = re.search(r"管理会社\s*([^\n]{2,80})", text)
    if manager:
        put("management_company", manager.group(1).strip())
    total = number(r"総戸数[^0-9]{0,20}([0-9,]+)\s*戸", int)
    if total:
        put("total_units", total)
    floor_pair = re.search(r"地上\s*([0-9]+)\s*階[^0-9]{0,20}([0-9]+)\s*階", flat, re.S)
    if floor_pair:
        put("total_floors", int(floor_pair.group(1)))
        put("unit_floor", floor_pair.group(2))
    return out

def _netloc_of(url: str) -> str:
    """Lower-cased network location of a URL, www-stripped; '' on garbage."""
    try:
        return urllib.parse.urlparse(url.lower().strip()).netloc.removeprefix("www.")
    except Exception:
        return ""

def _portal_for_netloc(netloc: str) -> str | None:
    """Map a netloc to a known portal by EXACT domain or true subdomain match.
    'kenbiya.com' and 'm.kenbiya.com' match Kenbiya; a spoof like
    'kenbiya.com.evil.example' or '?ref=kenbiya.com' in the query does NOT —
    the old substring check was vulnerable to both."""
    if not netloc:
        return None
    for portal, domain in PORTAL_DOMAINS.items():
        bare = domain.removeprefix("www.")
        if netloc == bare or netloc.endswith("." + bare):
            return portal
    return None

def parse_listing_url(url: str) -> tuple[dict, list[str]]:
    """OFFLINE URL-slug parser: extracts whatever the URL string itself encodes
    — portal (domain), prefecture (path slug), and for Rakumachi a property-
    type code — with ZERO network access, so it carries no scraping/ToS weight
    at all. Returns ({man_* updates}, [labels of what was filled])."""
    updates: dict = {}
    filled: list[str] = []
    # Cheap early-exit: random prose pasted in the box ("Check this property
    # out") isn't a URL; skip parsing entirely rather than letting urlparse
    # treat it as a path and walk segments for nothing.
    if not url or not url.lower().lstrip().startswith(("http://", "https://")):
        return updates, filled
    low = url.lower()
    portal = _portal_for_netloc(_netloc_of(url))
    if portal in MANUAL_PORTALS:
        updates["man_portal"] = portal
        filled.append(tr("form_portal"))
    try:
        segs = [p for p in urllib.parse.urlparse(low).path.split("/") if p]
    except Exception:
        segs = []
    for seg in segs:
        if seg in PREF_SLUG_MAP and "man_pref" not in updates:
            updates["man_pref"] = PREF_SLUG_MAP[seg]
            filled.append(tr("form_pref"))
        if seg in RAKUMACHI_DIM_PTYPE and "man_ptype" not in updates:
            updates["man_ptype"] = RAKUMACHI_DIM_PTYPE[seg]
            filled.append(tr("form_ptype"))
    # Once a prefecture is known, try to resolve a 市区町村コード from the path
    # slugs too (portal URLs often carry a city romaji segment, e.g. Kenbiya's
    # .../s/<pref>/<city>/...). resolve_municipality_code does an EXACT romaji
    # match, so an unrelated segment never yields a false city. Still offline.
    pref_for_city = updates.get("man_pref")
    if pref_for_city and "man_city_code" not in updates:
        for seg in segs:
            code = resolve_municipality_code(pref_for_city, seg)
            if code:
                updates["man_city_code"] = code
                filled.append(tr("form_city_label"))
                break
    return updates, filled

class _AllowlistRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validates EVERY redirect hop against the portal allowlist. urllib's
    default handler follows 301/302/307 blindly — so an open redirect on a
    portal (or a compromised one) could bounce the fetch to an internal target
    (169.254.169.254, localhost, a VPC host) AFTER the initial string check
    already passed. This closes that gap by running each redirect target back
    through the same scheme + _portal_for_netloc guard, raising rather than
    following anything off-allowlist."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        scheme = urllib.parse.urlparse((newurl or "").lower().strip()).scheme
        if scheme not in ("http", "https") or _portal_for_netloc(_netloc_of(newurl)) is None:
            raise VisionError("redirect to a non-approved destination blocked")
        return super().redirect_request(req, fp, code, msg, headers, newurl)

# Opener that (a) re-validates redirects and (b) caps redirect depth. Built once.
_SAFE_OPENER = urllib.request.build_opener(_AllowlistRedirectHandler())

def _fetch_page_text(url: str) -> str:
    """One user-initiated GET of the pasted listing page (isolated so tests can
    monkeypatch it). Raises on any failure; callers translate that into a
    friendly 'use the screenshot path' message."""
    # SSRF guard: only http(s), and only to the recognized portal domains (or
    # their true subdomains). Without this, a deployed instance would GET any
    # pasted string — including cloud metadata IPs (169.254.169.254), loopback,
    # or internal VPC hosts. The custom opener extends this same check to every
    # redirect hop, so a portal redirect can't smuggle the fetch off-allowlist.
    scheme = urllib.parse.urlparse(url.lower().strip()).scheme
    if scheme not in ("http", "https"):
        raise VisionError("only http/https URLs are fetched")
    if _portal_for_netloc(_netloc_of(url)) is None:
        raise VisionError("URL domain is not an approved portal")
    # No custom User-Agent: urllib sends its plain default (Python-urllib/x.y).
    # That's a truthful client identifier — NOT a browser disguise. The earlier
    # "jp-re-screener bot" UA string was what some portals' filters keyed on to
    # return 403; removing it restores the prior (working) behavior on portals
    # that tolerate a plain client, without pretending to be a browser. Portals
    # that still block (e.g. Rakumachi) are exercising their access policy —
    # use the screenshot path for those rather than disguising the request.
    req = urllib.request.Request(url, headers={"Accept-Language": "ja,en;q=0.8"})
    try:
        with _SAFE_OPENER.open(req, timeout=20) as resp:
            return resp.read(1_500_000).decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        # Portals like Rakumachi return 403/401 to non-browser clients by
        # policy — that's their access decision, not a fixable error. Signal it
        # distinctly so the UI can point the user to the screenshot path rather
        # than show a cryptic "HTTP error 403".
        if e.code in (401, 403, 429):
            raise PortalBlockedError(str(e.code)) from e
        raise

import re as _re
_TITLE_RE = _re.compile(r"<title[^>]*>(.*?)</title>", _re.S | _re.I)
# Price forms in the wild (post-NFKC, so half-width):
#   "6,290万円" | "1億2,000万円" | "2億円" | "1.5億円"
# The 億-aware pattern must run FIRST: the plain 万円 pattern alone would latch
# onto the "2,000万円" tail of "1億2,000万円" and silently score a ¥120M asset
# as ¥20M.
_PRICE_OKU_RE = _re.compile(r"([\d,.]+)\s*億\s*(?:([\d,]+)\s*万)?円")
_PRICE_MAN_RE = _re.compile(r"([\d,]+)\s*万円")
# Yield: prefer a value anchored to 利回り/表面 so occupancy figures like
# 「満室稼働率98%」can't masquerade as yield; fall back to the first bare %
# (Kenbiya titles carry an unanchored "4.00%"), with normalize_extraction's
# 0–30% gate as the final guard.
_YIELD_ANCHORED_RE = _re.compile(r"(?:利回り|表面)[^\d%]*([\d.]+)\s*%")
_YIELD_RE = _re.compile(r"([\d.]+)\s*%")

def _extract_from_title(html: str) -> dict:
    """Tier-1 page extraction, NO API key needed: Kenbiya (and others) put the
    asking price and gross yield straight into the <title> tag, e.g.
    「習志野市 6,290万円 4.00% 区分マンション…」.

    The title is NFKC-normalized first. The subtle full-width bug this kills:
    regex digit classes match full-width digits and float() even accepts them
    bare, but full-width PUNCTUATION (，decimal-point ．) makes float() raise,
    so safe_float silently returns NaN and the price vanishes
    with no error. NFKC folds ６，２９０→6,290 and ％→% before the regexes run,
    so 「６，２９０万円 ４．００％」extracts identically to the half-width form."""
    m = _TITLE_RE.search(html)
    if not m:
        return {}
    title = unicodedata.normalize("NFKC", m.group(1))
    raw: dict = {}
    po = _PRICE_OKU_RE.search(title)
    if po:
        oku = safe_float(po.group(1).replace(",", ""), 0.0)
        man = safe_float((po.group(2) or "0").replace(",", ""), 0.0)
        raw["price_man"] = oku * 10_000 + man
    else:
        pm = _PRICE_MAN_RE.search(title)
        if pm:
            raw["price_man"] = safe_float(pm.group(1).replace(",", ""))
    ym = _YIELD_ANCHORED_RE.search(title) or _YIELD_RE.search(title)
    if ym:
        raw["gross_yield_pct"] = safe_float(ym.group(1))
    return raw

def _strip_html(html: str, limit: int = 30000) -> str:
    """Strip a listing page to LLM-ready text. When the page exceeds the budget,
    don't guess where the data lives — LOOK for it: if a spec-table anchor
    (物件概要 / 専有面積 / 用途地域 / 利回り) is present, keep the page head plus
    a window CENTERED on the first anchor, guaranteeing the 物件概要 table
    survives no matter how much nav precedes it or footer follows it. Only when
    no anchor exists fall back to a head+tail split (better than either end
    alone). The 30k budget means truncation rarely triggers at all."""
    text = _re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=_re.S | _re.I)
    text = _re.sub(r"<[^>]+>", " ", text)
    text = _re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    head = limit // 5
    for anchor in ("物件概要", "専有面積", "用途地域", "利回り"):
        idx = text.find(anchor)
        if idx > head:
            win = limit - head
            start = max(head, idx - win // 3)
            return text[:head] + " … " + text[start:start + win]
    return text[:head] + " … " + text[-(limit - head):]

def extract_listing_from_url(url: str) -> dict:
    """Tier-1 + Tier-2 page extraction. Tier 1 regexes price/yield out of the
    <title> (works with no API key). Tier 2, when a real LLM provider is
    configured (anything but 'mock'), sends the stripped page text through the
    same _VISION_PROMPT contract for the remaining fields. Output feeds the
    same normalize_extraction trust boundary as screenshots — page content is
    untrusted input either way."""
    html = _fetch_page_text(url)
    raw = _extract_from_title(html)
    provider = (_get_secret("VISION_PROVIDER", "mock") or "mock").lower()
    if provider in ("openai", "gemini", "anthropic"):
        try:
            prompt = _VISION_PROMPT + "\n\nPage text:\n" + _strip_html(html)
            if provider == "anthropic":
                llm_raw = _text_llm_anthropic(prompt)
            elif provider == "openai":
                llm_raw = _text_llm_openai(prompt)
            else:
                llm_raw = _text_llm_gemini(prompt)
            llm_raw.update({k: v for k, v in raw.items() if v is not None})  # title regex wins on conflict
            raw = llm_raw
        except Exception:
            pass   # tier-2 is best-effort; tier-1 results still apply
    if not raw:
        raise VisionError("no extractable data (page may be login-walled)")
    return raw

def _text_llm_openai(prompt: str) -> dict:
    key = _get_secret("OPENAI_API_KEY")
    if not key:
        raise VisionError("OPENAI_API_KEY not set")
    body = json.dumps({"model": _get_secret("VISION_MODEL_OPENAI", "gpt-4o"),
                       "max_tokens": 500,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    with _urlopen_with_backoff(req, timeout=60, retries=4) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return _json_from_text(payload["choices"][0]["message"]["content"])

def _text_llm_gemini(prompt: str) -> dict:
    key = _get_secret("GEMINI_API_KEY")
    if not key:
        raise VisionError("GEMINI_API_KEY not set")
    model = _get_secret("VISION_MODEL_GEMINI", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json",
                                          "x-goog-api-key": key})
    with _urlopen_with_backoff(req, timeout=60, retries=4) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return _json_from_text(payload["candidates"][0]["content"]["parts"][0]["text"])

def _text_llm_anthropic(prompt: str) -> dict:
    key = _get_secret("ANTHROPIC_API_KEY")
    if not key:
        raise VisionError("ANTHROPIC_API_KEY not set")
    body = json.dumps({"model": _get_secret("VISION_MODEL_ANTHROPIC", "claude-sonnet-4-6"),
                       "max_tokens": 500,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                          "Content-Type": "application/json"})
    with _urlopen_with_backoff(req, timeout=60, retries=4) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return _json_from_text("".join(b.get("text", "") for b in payload.get("content", [])))

# ----------------------------------------------------------------------------
# Tab 3 — Analyze Property (manual entry of real Kenbiya / Rakumachi listings)
# ----------------------------------------------------------------------------
_MAN_FORM_DEFAULTS = {
    "man_portal": "Kenbiya", "man_url": "", "man_ptype": PROPERTY_TYPES[0],
    # Neutral first-use values. These previously held sample figures (3,000万円,
    # 25m², 6.5%, 8 min, 20 yrs) which look exactly like extracted facts and
    # could be scored unnoticed. Station and age keep numeric widgets but are
    # marked UNKNOWN by default — zeroing them would read as "0-minute walk,
    # brand new", which is worse than a demo value, not better.
    "man_pref": PREFECTURES[0], "man_price": 0, "man_area": 0.0,
    "man_station": 0, "man_age": 0, "man_yield": 0.0, "man_fees": 0,
    "man_station_unknown": True, "man_age_unknown": True,
    "man_yield_basis": YIELD_BASIS_DEFAULT, "man_nickname": "",
    "man_building_name": "", "man_unit_number": "", "man_unit_floor": "",
    "man_source_document_name": "", "man_source_document_date": "",
    "man_ingestion_channel": "manual", "man_source_type": "unknown",
    "man_agency_name": "", "man_agent_name": "", "man_agency_phone": "",
    "man_agency_address": "", "man_agency_role": "unknown", "man_seller_name": "",
    "man_management_company": "", "man_rent_guarantee_company": "",
    "man_monthly_rent": 0, "man_market_rent": 0, "man_occupancy": "unknown",
    "man_structure": "Unknown", "man_extracted_entities": [], "man_conflicts": [],
    "man_pasted_text": "", "man_paste_warnings": [], "man_paste_fieldcount": 0,
    "man_provenance": {},
    "man_zoning": "unknown", "man_far": 0, "man_track": True,
    "man_down_payment": 20, "man_interest_rate": 2.0, "man_loan_years": 25, "man_target_coc": 5.0,
    # Resolved 5-digit 市区町村コード (set by URL/screenshot extraction; None ->
    # the scoring engine uses the prefecture benchmark). No widget binds to it.
    "man_city_code": None, "man_address": "", "man_coords": None,
}

def _ensure_manual_form_state() -> None:
    """Seed every form key once. Widgets below carry NO value=/index= params, so
    session_state is the single source of truth — which is what lets the vision
    extractor (which runs BEFORE the form instantiates) pre-fill the fields
    instantly within the same run, with no rerun and no Streamlit
    'value vs session_state' warnings."""
    for k, v in _MAN_FORM_DEFAULTS.items():
        st.session_state.setdefault(k, v)

def render_manual_analyzer() -> None:
    st.markdown(f"### {tr('analyze_header')}")
    st.caption(tr("analyze_intro"))
    _ensure_manual_form_state()

    # --- URL import: renders before the form for the same session-state-
    # ordering reason as the screenshot section below. ------------------------
    st.markdown(f"##### {tr('url_import_header')}")
    url_in = st.text_input(tr("form_url"), key="man_url",
                           placeholder="https://www.kenbiya.com/pp1/s/...")
    # Auto-parse the URL slug the moment it changes (offline, instant). The
    # _last_parsed_url guard means we fill ONCE per pasted URL — subsequent
    # reruns never clobber the user's manual corrections to those fields.
    if url_in and url_in != st.session_state.get("_last_parsed_url"):
        st.session_state["_last_parsed_url"] = url_in
        upd, filled = parse_listing_url(url_in)
        for k, v in upd.items():
            st.session_state[k] = v
        if filled:
            st.info(tr("url_autofill_info", items=", ".join(filled)))
    st.caption(tr("url_fetch_note"))
    if url_in and st.button(tr("url_fetch_btn"), width="stretch"):
        try:
            # Per-URL result cache: re-clicking for the same URL reuses the
            # parsed result instead of re-hitting the page and the LLM.
            cache = st.session_state.setdefault("_url_fetch_cache", {})
            if url_in in cache:
                raw = cache[url_in]
            else:
                with st.spinner(tr("url_fetch_spinner")):
                    raw = extract_listing_from_url(url_in)
                cache[url_in] = raw
            updates, warns = normalize_extraction(raw)
            if "man_address" in updates:
                st.session_state["man_coords"] = None   # new address -> stale coords
            for k, v in updates.items():
                st.session_state[k] = v
            st.success(tr("extract_success", n=len(updates)))
            for key, kwargs in warns:
                st.warning(tr(key, **kwargs))
        except PortalBlockedError:
            # Surface what the offline slug parse already filled, so the user
            # sees partial progress rather than a dead end, then point them to
            # the screenshot path that bypasses the block entirely.
            already = parse_listing_url(url_in)[1]
            if already:
                st.warning(tr("url_fetch_blocked_partial", items=", ".join(already)))
            else:
                st.warning(tr("url_fetch_blocked"))
        except Exception as e:   # login wall, parse failure, network
            st.error(tr("url_fetch_failed", msg=_scrub_secrets(str(e))[:120]))

    # --- Browser-extension paste: ingest a listing the user captured via the
    # local extension (manual, logged-in, their own page view). Same trust
    # boundary as every other path. Renders before the form for write ordering.
    with st.expander(tr("ext_header")):
        st.caption(tr("ext_help"))
        ext_text = st.text_area(tr("ext_paste_label"), key="ext_paste", height=120,
                                placeholder='{"price_man": 6290, "area_sqm": 25.3, ...}')
        if st.button(tr("ext_ingest_btn"), width="stretch") and ext_text.strip():
            try:
                raw = parse_extension_payload(ext_text)
                updates, warns = normalize_extraction(raw)
                if "man_address" in updates:
                    st.session_state["man_coords"] = None
                for k, v in updates.items():
                    st.session_state[k] = v
                st.success(tr("extract_success", n=len(updates)))
                for key, kwargs in warns:
                    st.warning(tr(key, **kwargs))
            except Exception as e:
                st.error(tr("ext_ingest_failed", msg=_scrub_secrets(str(e))[:160]))

    # --- Screenshot ingestion: MUST render before the form so its writes to
    # the man_* keys land before those widgets instantiate this run. ---------
    st.markdown(f"##### {tr('upload_header')}")
    st.caption(tr("vision_note"))
    # png/jpeg only: every provider accepts these; Gemini is historically strict
    # about mime_type matching the payload, and webp support varies by model.
    # PDF is accepted alongside images. A text PDF is read directly (no vision
    # call); scanned pages are rendered and sent to the provider. Multi-page
    # documents merge through merge_extractions(), and CONFLICTS ARE SHOWN, not
    # silently resolved. Uploaded bytes live only for this call — nothing is
    # written to disk or the database.
    # --- Select All / Copy route ---------------------------------------------
    # For portals that block server-side fetching. Nothing is requested from the
    # portal: the person copies the page they are already looking at. The text
    # is untrusted, so it only PREFILLS the review form below — it never scores
    # or saves anything by itself.
    with st.expander(tr("paste_header")):
        st.caption(tr("paste_caption"))
        pasted = st.text_area(tr("paste_label"), height=320, key="man_pasted_text")
        if st.button(tr("paste_extract_btn"), width="stretch"):
            if not (pasted or "").strip():
                st.warning(tr("paste_empty"))
            else:
                fields = parse_pasted_listing(pasted)
                updates, warns = normalize_extraction(fields)
                if "man_address" in updates:
                    st.session_state["man_coords"] = None
                for k, v in updates.items():
                    st.session_state[k] = v
                # Field-level provenance, not the page. The pasted text itself is
                # NOT persisted unless the user explicitly opts in below.
                st.session_state["man_provenance"] = {
                    k: {"extracted_value": v, "source_document": None,
                        "page_number": None, "extraction_method": "pasted_text",
                        "confidence": None}
                    for k, v in fields.items() if not isinstance(v, (list, dict))}
                st.session_state["man_paste_warnings"] = pasted_listing_warnings(fields)
                st.session_state["man_paste_fieldcount"] = len(
                    [v for v in fields.values() if v not in (None, "", [], {})])
                st.success(tr("paste_extracted",
                              n=st.session_state["man_paste_fieldcount"]))
                st.rerun()
        if st.session_state.get("man_paste_fieldcount"):
            st.caption(tr("paste_review_note",
                          n=st.session_state["man_paste_fieldcount"]))
        for w in st.session_state.get("man_paste_warnings") or []:
            st.warning(tr(w["key"], **w["kwargs"]))
        st.caption(tr("paste_not_stored"))

    shots = st.file_uploader(tr("upload_label"), type=["pdf", "png", "jpg", "jpeg"],
                             accept_multiple_files=True, key="man_shot")
    if shots and st.button(tr("extract_btn"), width="stretch"):
        per_page, warns_doc, doc_name = [], [], None
        try:
            with st.spinner(tr("extract_spinner")):
                for up in shots:
                    doc_name = doc_name or up.name
                    data = up.getvalue()
                    parsed = extract_document_pages(data, up.name)
                    warns_doc += parsed.get("warnings", [])
                    for pg in parsed["pages"]:
                        if pg.get("image_b64"):
                            raw = extract_listing_from_image(
                                base64.b64decode(pg["image_b64"]), "image/png")
                            meth = "vision"
                        elif pg.get("text"):
                            # Deterministic recovery runs on every text page,
                            # even when a vision provider is configured: for
                            # labelled numbers in an embedded text layer it is
                            # both cheaper and more reliable than a model.
                            raw = recover_document_fields(
                                extract_listing_from_text(pg["text"]), pg["text"])
                            meth = "text"
                        else:
                            continue
                        per_page.append({"page": pg["page"], "method": meth,
                                         "document": up.name, "fields": raw or {}})
            merged, conflicts, provenance = merge_extractions(per_page)
            st.session_state["man_provenance"] = provenance
            # Company/role pairs -> the right FIELD. An unlabelled company never
            # becomes the agency; it stays in extracted_entities for review.
            pairs = merged.pop("company_roles", []) or []
            if pairs:
                merged.update(map_company_roles([tuple(x) for x in pairs]))
            channel = ("agent_pdf" if any(f.name.lower().endswith(".pdf") for f in shots)
                       else "agent_image")
            merged.setdefault("ingestion_channel", channel)
            merged.setdefault("source_document_name", doc_name)
            updates, warns = normalize_extraction(merged)
            if (_get_secret("VISION_PROVIDER", "mock") or "mock").lower() == "mock":
                warns.insert(0, ("warn_mock_provider", {}))
            if "man_address" in updates:
                st.session_state["man_coords"] = None   # new address -> stale coords
            for k, v in updates.items():
                st.session_state[k] = v
            st.session_state["man_conflicts"] = conflicts
            st.success(tr("extract_success", n=len(updates)))
            for w in warns_doc:
                st.warning(w)
            for key, kwargs in warns:
                st.warning(tr(key, **kwargs))
        except DocumentError as e:
            st.error(str(e))
        except Exception as e:   # VisionError or any provider/network failure
            st.error(vision_error_message(e))

    # Conflicts must be reviewed before they can influence anything. The values
    # sit in the form either way — this is the prompt to check them.
    _conf = st.session_state.get("man_conflicts") or []
    if _conf:
        st.warning(tr("doc_conflict_header"))
        for cf in _conf:
            st.write("• " + tr("doc_conflict_row", field=cf["field"], kept=cf["kept"],
                               other=cf["other"], page=cf["page_number"]))
        if st.button(tr("doc_conflict_ack"), width="stretch"):
            st.session_state["man_conflicts"] = []
            st.rerun()

    # --- Source attribution: provenance and workflow, NEVER a score input -----
    with st.expander(tr("attrib_header")):
        st.caption(tr("attrib_caption"))
        a1, a2 = st.columns(2)
        with a1:
            st.selectbox(tr("form_ingestion_channel"), INGESTION_CHANNELS,
                         format_func=lambda k: tr(f"channel_{k}"), key="man_ingestion_channel")
            st.text_input(tr("form_agency_name"), key="man_agency_name",
                          placeholder=tr("agency_not_identified"))
            st.text_input(tr("form_agent_name"), key="man_agent_name")
            st.text_input(tr("form_agency_phone"), key="man_agency_phone")
            st.text_input(tr("form_seller_name"), key="man_seller_name")
        with a2:
            st.selectbox(tr("form_source_type"), SOURCE_TYPES,
                         format_func=lambda k: tr(f"srctype_{k}"), key="man_source_type")
            st.selectbox(tr("form_agency_role"), AGENCY_ROLES,
                         format_func=lambda k: tr(f"arole_{k}"), key="man_agency_role")
            st.text_input(tr("form_management_company"), key="man_management_company")
            st.text_input(tr("form_rent_guarantee"), key="man_rent_guarantee_company")
            st.text_input(tr("form_source_doc"), key="man_source_document_name")
        ents = st.session_state.get("man_extracted_entities") or []
        if ents:
            st.warning(tr("attrib_unknown_entities",
                          items=", ".join(str(e.get("name", "?")) for e in ents)))

    # --- Address + 📍 Locate: outside the form so the Locate button can run a
    # geocode immediately. Local (VS Code) runs auto-fill this from the page
    # fetch; online/mobile users paste or type the address, then tap Locate. A
    # button (not on-change) means a half-typed address never fires a geocode.
    st.markdown(f"##### {tr('address_header')}")
    st.caption(tr("address_tip"))
    ac1, ac2 = st.columns([4, 1])
    with ac1:
        address = st.text_input(tr("form_address"), key="man_address",
                                help=tr("form_address_help"), label_visibility="collapsed")
    with ac2:
        locate = st.button(tr("locate_btn"), width="stretch")
    if locate and (st.session_state.get("man_address") or "").strip():
        addr = st.session_state["man_address"].strip()
        # Recognising which ward an address is in is a STATIC table lookup. It
        # does not need geocoding, and it certainly does not need an MLIT
        # benchmark refresh — that only supplies transaction data for a ward we
        # have already identified. Resolve first so a geocoder failure below
        # cannot erase a ward we correctly recognised.
        resolved = resolve_municipality_code(st.session_state.get("man_pref"), addr)
        if resolved:
            st.session_state["man_city_code"] = resolved
        coords = geocode_address(addr)
        st.session_state["man_coords"] = coords
        st.session_state["man_coords_addr"] = addr   # remember WHAT we geocoded
        if coords:
            geomock = (_get_secret("GEOCODER", "gsi") or "gsi").lower() == "mock"
            st.caption(tr("geocode_mock" if geomock else "geocode_ok",
                          lat=coords[0], lon=coords[1]))
        else:
            st.caption(tr("geocode_miss"))

    with st.form("manual_analyze"):
        c1, c2 = st.columns(2)
        with c1:
            portal = st.selectbox(tr("form_portal"), MANUAL_PORTALS, key="man_portal")
            ptype = st.selectbox(tr("form_ptype"), PROPERTY_TYPES, format_func=ptype_name, key="man_ptype")
            pref = st.selectbox(tr("form_pref"), PREFECTURES, format_func=pref_name, key="man_pref")
        with c2:
            price_man = st.number_input(tr("form_price"), min_value=0, step=100, key="man_price")
            area = st.number_input(tr("form_area"), min_value=0.0, step=0.5, key="man_area")
            # Explicit unknown beats a silent default. When ticked the value is
            # passed as NaN, which _band_score maps to a neutral 50 — neither
            # rewarded nor punished — and the field is listed as missing evidence.
            # Unknown means "no evidence yet", NOT "you may not type here". The
            # previous version disabled the input, so a document that missed the
            # station left the user unable to correct it at all — a checkbox that
            # locks you out of the fix is worse than no checkbox.
            _st_unknown = st.checkbox(tr("form_station_unknown"), key="man_station_unknown")
            station = st.number_input(tr("form_station"), min_value=0, max_value=60,
                                      key="man_station")
            _age_unknown = st.checkbox(tr("form_age_unknown"), key="man_age_unknown")
            age = st.number_input(tr("form_age"), min_value=0, max_value=80,
                                  key="man_age")
            # Typing a value is itself evidence: it wins over the stale flag.
            # The resolution happens in the LOCAL variable only — Streamlit
            # forbids assigning st.session_state[k] once the widget with key k
            # has been instantiated, and writing back served no purpose anyway
            # since the value below is what the score consumes.
            if _st_unknown and station <= 0:
                station = float("nan")
            if _age_unknown and age <= 0:
                age = float("nan")
            # Contractual rent alone is sufficient for NOI, so the advertised
            # yield is optional. Left at 0 with a rent entered, the implied yield
            # is computed from the lease instead.
            gy = st.number_input(tr("form_yield_optional"), min_value=0.0, max_value=30.0,
                                 step=0.1, help=tr("form_yield_optional_help"), key="man_yield")
            _rent_now = safe_float(st.session_state.get("man_monthly_rent"))
            _price_now = safe_float(st.session_state.get("man_price"))
            if gy <= 0 and _rent_now > 0 and _price_now > 0:
                st.caption(tr("form_yield_implied",
                              pct=f"{_rent_now * 12 / (_price_now * 10_000) * 100:.2f}"))
            fees = st.number_input(tr("form_fees"), min_value=0, max_value=500_000,
                                   step=1000, help=tr("form_fees_help"), key="man_fees")
            nickname = st.text_input(tr("form_nickname"), max_chars=60, help=tr("form_nickname_help"), key="man_nickname")
            bname = st.text_input(tr("form_building_name"), max_chars=120, key="man_building_name")
            uc1, uc2 = st.columns(2)
            with uc1:
                unit_no = st.text_input(tr("form_unit_number"), max_chars=20, key="man_unit_number")
            with uc2:
                unit_fl = st.text_input(tr("form_unit_floor"), max_chars=10, key="man_unit_floor")
            structure = st.selectbox(tr("form_structure"), ["Unknown","RC","SRC","S","Wood"], key="man_structure")
            # Contractual rent is the base case. Market rent is upside ONLY, held
            # in its own field — it never feeds the base-case NOI.
            rc1, rc2 = st.columns(2)
            with rc1:
                rent_contract = st.number_input(tr("rent_contract_label"), min_value=0,
                                                max_value=10_000_000, step=1000, key="man_monthly_rent")
            with rc2:
                rent_market = st.number_input(tr("rent_market_label"), min_value=0,
                                              max_value=10_000_000, step=1000,
                                              help=tr("rent_market_help"), key="man_market_rent")
            occupancy = st.selectbox(tr("form_occupancy"), ["unknown","rented","vacant"], key="man_occupancy")
            ybasis = st.selectbox(tr("form_yield_basis"), YIELD_BASES,
                                  format_func=yield_basis_name,
                                  help=tr("form_yield_basis_help"), key="man_yield_basis")
        z1, z2 = st.columns(2)
        with z1:
            zoning = st.selectbox(tr("form_zoning"), ZONING_KEYS,
                                  format_func=zoning_name, help=tr("form_zoning_help"), key="man_zoning")
        with z2:
            far_pct = st.number_input(tr("form_far"), min_value=0, max_value=1300,
                                      step=10, help=tr("form_far_help"), key="man_far")
        track = st.checkbox(tr("form_track"), key="man_track")
        submitted = st.form_submit_button(tr("analyze_btn"), width="stretch", type="primary")

    if not submitted:
        return

    url = st.session_state.get("man_url", "")   # URL now lives in the import section
    price_yen = float(price_man) * 10_000.0   # 万円 -> ¥, matching JP listing convention
    if price_yen <= 0 or area <= 0 or (gy <= 0 and not rent_contract):
        st.warning(tr("invalid_input"))
        return
    if url and PORTAL_DOMAINS.get(portal, "") not in url:
        st.warning(tr("url_mismatch", portal=portal))

    # Use the city_code resolved during URL/screenshot extraction, but only if
    # it still belongs to the prefecture currently selected in the form — the
    # user may have changed the prefecture after an extraction, which would
    # otherwise leave a stale ward code attached to the wrong prefecture.
    city_code = st.session_state.get("man_city_code") or None
    if city_code and _pref_key_from_city_code(city_code) != pref:
        city_code = None

    # Coordinates: prefer what 📍 Locate already resolved (cached in session);
    # otherwise geocode now, so a user who filled the address but skipped Locate
    # still gets a map pin + hazard check. None on any miss — never blocks scoring.
    address = (st.session_state.get("man_address") or "").strip()
    coords = st.session_state.get("man_coords")
    # Discard cached coords if the address was edited (typed over) since Locate —
    # otherwise we'd persist the previous listing's lat/lon and run the hazard
    # check against the wrong location.
    if coords is not None and st.session_state.get("man_coords_addr") != address:
        coords = None
    if coords is None and address:
        coords = geocode_address(address)

    # Deterministic ID from the URL (or the field tuple when no URL given) so
    # re-analyzing the same listing on the same day refreshes, never duplicates.
    ident = url or f"{portal}|{pref}|{ptype}|{price_yen}|{area}"
    listing_id = "MAN-" + hashlib.sha256(ident.encode()).hexdigest()[:8].upper()
    row = {
        "listing_id": listing_id,
        "url": url or f"https://{PORTAL_DOMAINS.get(portal, 'example.com')}/",
        "portal": portal,
        "title": f"{ptype} · {area}m2 · {pref} · {station}min walk · {age}yrs",
        "ptype": ptype, "prefecture": pref, "city_code": city_code,
        "area_sqm": float(area), "station_min": float(station),
        "building_age": float(age), "price_yen": price_yen,
        "gross_yield": float(gy), "monthly_fees_yen": float(fees) if fees else None,
        "yield_basis": ybasis, "nickname": (nickname or "").strip() or None,
        # WS3 identity / WS5 attribution / rent split — carried onto the scored
        # row so save_manual_pick() persists exactly what the user reviewed.
        "building_name": (bname or "").strip() or None,
        "unit_number": (unit_no or "").strip() or None,
        "unit_floor": (unit_fl or "").strip() or None,
        "structure": structure if structure != "Unknown" else None,
        "occupancy": occupancy if occupancy != "unknown" else None,
        "monthly_rent_yen": float(rent_contract) if rent_contract else None,
        "market_rent_yen": float(rent_market) if rent_market else None,
        "source_document_name": (st.session_state.get("man_source_document_name") or "").strip() or None,
        "source_document_date": (st.session_state.get("man_source_document_date") or "").strip() or None,
        "ingestion_channel": st.session_state.get("man_ingestion_channel") or "manual",
        "source_type": st.session_state.get("man_source_type") or "unknown",
        "agency_name": (st.session_state.get("man_agency_name") or "").strip() or None,
        "agent_name": (st.session_state.get("man_agent_name") or "").strip() or None,
        "agency_phone": (st.session_state.get("man_agency_phone") or "").strip() or None,
        "agency_address": (st.session_state.get("man_agency_address") or "").strip() or None,
        "agency_role": st.session_state.get("man_agency_role") or "unknown",
        "seller_name": (st.session_state.get("man_seller_name") or "").strip() or None,
        "management_company": (st.session_state.get("man_management_company") or "").strip() or None,
        "rent_guarantee_company": (st.session_state.get("man_rent_guarantee_company") or "").strip() or None,
        "extracted_entities": st.session_state.get("man_extracted_entities") or None,
        "zoning": zoning, "far_pct": far_or_nan(far_pct),
        "lat": coords[0] if coords else None,
        "lon": coords[1] if coords else None,
        "status": "Active",
    }
    r = analyze_properties(pd.DataFrame([row])).iloc[0]

    # Say plainly what was unknown, so a neutral-by-ignorance score is never
    # mistaken for a neutral-by-measurement one.
    _missing = [tr("form_station") for _ in (1,) if math.isnan(safe_float(station))]
    _missing += [tr("form_age") for _ in (1,) if math.isnan(safe_float(age))]
    if _missing:
        st.info(tr("missing_evidence", items=", ".join(_missing)))

    # NOI is computed ONCE here, immediately after scoring, and the same object
    # is reused for persistence, the headline yield card, the NOI explanation and
    # the financing scenario. It previously lived further down the function while
    # the persistence block above referenced it — an UnboundLocalError that fired
    # the moment anyone scored a property. Single assignment, single source.
    _ny = compute_net_yield(price_yen, float(r["gross_yield"]), float(r["area_sqm"]),
                            r.get("ptype"), r.get("monthly_fees_yen"),
                            r.get("yield_basis") or YIELD_BASIS_DEFAULT,
                            monthly_rent_yen=r.get("monthly_rent_yen"),
                            market_rent_yen=r.get("market_rent_yen"))
    _net = _ny["net_yield_pct"]

    # --- Hazard gate: a flood/landslide zone is a categorical risk, so it
    # applies a fixed penalty to the displayed score and a prominent banner —
    # NOT a smooth sixth factor. Runs only with coordinates + an MLIT key; a
    # miss yields no hazards and leaves the score untouched (never a false safe).
    hazards = check_hazards(coords[0], coords[1]) if coords else {}
    flagged = [hz for hz, info in hazards.items() if info.get("hit")]
    dev = check_development(coords[0], coords[1]) if coords else {}
    dev_flagged = [k for k, hit in dev.items() if hit]
    base_score = float(r["score"])
    # Gate adjustments, both user-tunable in the KPI Weights tab:
    #   hazard  -> slider penalty x the WORST zone's severity multiplier, so a
    #              yellow <0.5m flood fringe (~0.5x) is not punished like a
    #              deep-red 10m basin (1.5x) or a 特別警戒 landslide zone (1.3x)
    #   develop -> bonus PER flag, capped at 2 (kodo + road both firing is the
    #              strongest official redevelopment signal we can read)
    max_sev = max((hazards[hz].get("sev", 1.0) for hz in flagged), default=0.0)
    hazard_pen = get_hazard_penalty() * max_sev if flagged else 0.0
    # Tiered bonus: being INSIDE a zone / on a planned road is a full flag;
    # 'zone one block away' and 'unusually active area' are half flags. Total
    # capped at 2 full flags, so max bonus stays dev_bonus x 2.
    DEV_FLAG_WEIGHTS = {"kodo": 1.0, "road": 1.0, "chiku": 1.0,
                        "kodo_near": 0.5, "chiku_near": 0.5, "activity": 0.5}
    dev_units = min(sum(DEV_FLAG_WEIGHTS.get(k, 0.5) for k in dev_flagged), 2.0)
    dev_bon = get_dev_bonus() * dev_units
    # Resale outlook: analyze_properties already applied the BASE floor-area
    # penalty (scans have no coordinates, so they can't consult demographics).
    # With coordinates we can, so scale that penalty by the projected population
    # trend and apply only the DELTA here — a compact unit in a shrinking
    # catchment compounds, the same unit in a growing one is discounted.
    outlook = check_population_outlook(coords[0], coords[1]) if coords else {}
    _liq_base = area_liquidity(r.get("area_sqm"), r.get("ptype"))
    resale_mult = resale_multiplier(outlook)
    # Demographic scaling belonged to the legacy area penalty. Applying it here
    # in evidence_weighted/descriptive mode would reintroduce the binary model.
    _mode = get_size_model_mode()
    # LEGACY ONLY: demographic scaling OF THE AREA PENALTY. Outside legacy mode
    # this stays 0.0, so the binary area model cannot return through the back door.
    resale_delta = (_liq_base["penalty"] * (resale_mult - 1.0)
                    if _mode == "legacy_penalty" else 0.0)
    # ALL MODES: a small standalone demographic gate that is independent of floor
    # area, so it double-counts nothing. Zero unless the outlook is determined.
    resale_gate = resale_gate_adjustment(outlook) if _mode != "legacy_penalty" else 0.0
    display_score = max(0.0, min(100.0, base_score - hazard_pen + dev_bon
                                - resale_delta + resale_gate))
    # Persist what the user actually SAW: the adjusted score and the flags that
    # caused it, plus the resolved city_code (so a re-score reproduces the same
    # ward benchmark). Without this the ledger and the verdict disagree.
    r["score"] = display_score
    r["city_code"] = city_code
    # Net yield and 旧耐震 are already computed inside analyze_properties (so
    # scans get them too); carry them onto the persisted row so the ledger keeps
    # the net figure and seismic class the user actually saw.
    r["net_yield"] = safe_float(r.get("net_yield"))
    r["monthly_fees_yen"] = float(fees) if fees else None
    r["seismic"] = r.get("seismic") or seismic_class(r.get("building_age"))[0]
    r["yield_basis"] = r.get("yield_basis") or YIELD_BASIS_DEFAULT
    r["net_yield_invested"] = safe_float(r.get("net_yield_invested"))
    # Carried from the scored row — NOT recomputed. analyze_properties already
    # produced this from the same inputs; recomputing here is how the panel and
    # the score drift apart.
    try:
        _sz = json.loads(r["size_breakdown"]) if r.get("size_breakdown") else {}
    except Exception:
        _sz = {}
    if not _sz:
        _sz = {"applicable": False, "final": float("nan"), "prior": float("nan"),
               "local": float("nan"), "evidence": 0.0, "functional": float("nan"),
               "rental": float("nan"), "exit": float("nan"), "n": 0,
               "similarity": 0.0, "missing": [], "rent_locally_benchmarked": False,
               "rent_source_kind": "none"}
    _size_adj_applied = safe_float(r.get("size_adjustment"), 0.0)
    r["size_resilience"] = _sz.get("final") if _sz.get("applicable") else None
    # Everything needed to reproduce this decision later, straight from the row.
    r["size_model_mode"] = r.get("size_model_mode") or get_size_model_mode()
    r["size_evidence"] = _sz.get("evidence")
    r["size_breakdown"] = (r.get("size_breakdown")
                           or (json.dumps(_sz, ensure_ascii=False, default=str) if _sz else None))
    # Stamp any field the user changed since extraction. Pass the DICT —
    # save_manual_pick owns serialisation; handing it a string double-encodes.
    r["provenance"] = apply_manual_overrides(
        st.session_state.get("man_provenance") or {}, r) or None
    r["rent_source"] = _ny.get("rent_source")
    r["contractual_gross_pct"] = _ny.get("contractual_gross_pct")
    r["gross_gap_pp"] = _ny.get("gross_gap_pp")
    r["area_tier"] = r.get("area_tier") or area_liquidity(r.get("area_sqm"), r.get("ptype"))["tier"]
    r["pop_outlook_pct"] = outlook.get("pct_change") if outlook else None
    # Persist flags WITH their grading (e.g. "flood:rank4,landslide:class2") so
    # the ledger records not just that a zone hit but how severe it was.
    r["hazard_flags"] = ",".join(
        f"{hz}:{hazards[hz]['detail']}" if hazards[hz].get("detail") else hz
        for hz in flagged) if flagged else None
    r["dev_flags"] = ",".join(dev_flagged) if dev_flagged else None

    # --- Risk strip -----------------------------------------------------------
    # Previously this was up to seven stacked full-width banners, which buried
    # the real signal in noise. Now: one scannable row of state chips, with the
    # full explanations one click away in a single expander that opens by itself
    # whenever something genuinely needs reading.
    sep = "、" if get_lang() == "ja" else ", "
    chips: list[tuple[str, str, str, str]] = []
    details: list[tuple[str, str]] = []          # (severity, message)

    # Zoning trap
    if zoning in ("choseikuiki", "kogyo_senyo"):
        chips.append(("🚧", tr("risk_zoning"), zoning_name(zoning), "risk"))
        details.append(("risk", tr("zoning_warning")))

    # Hazard
    if flagged:
        def _hz_name(hz: str) -> str:
            base = tr(f"hazard_{hz}")
            det = hazards[hz].get("detail")
            return f"{base} ({tr('hzsev_' + det)})" if det else base
        names = sep.join(_hz_name(hz) for hz in flagged)
        worst = sep.join(tr("hzsev_" + hazards[hz]["detail"]) if hazards[hz].get("detail")
                         else tr(f"hazard_{hz}") for hz in flagged)
        chips.append(("🌊", tr("risk_hazard"), worst, "risk"))
        details.append(("risk", tr("hazard_warning", zones=names,
                                   penalty=int(round(hazard_pen)))))
    elif hazards:
        chips.append(("🌊", tr("risk_hazard"), tr("rv_clear"), "ok"))
    else:
        chips.append(("🌊", tr("risk_hazard"), tr("rv_unchecked"), "unknown"))

    # 新耐震 / 旧耐震
    _scls, _syear = seismic_class(r.get("building_age"))
    if _scls == "old":
        chips.append(("🏚️", tr("risk_seismic"), tr("rv_old_seismic", year=_syear), "risk"))
        details.append(("risk", tr("seismic_old", year=_syear,
                                   penalty=int(round(get_old_seismic_penalty())))))
    elif _scls == "grey":
        chips.append(("🏚️", tr("risk_seismic"), tr("rv_grey_seismic", year=_syear), "caution"))
        details.append(("caution", tr("seismic_grey", year=_syear)))
    elif _scls == "new":
        chips.append(("🏚️", tr("risk_seismic"), tr("rv_new_seismic", year=_syear), "ok"))
    else:
        chips.append(("🏚️", tr("risk_seismic"), tr("rv_unchecked"), "unknown"))

    # Exit liquidity by floor area
    _liq = area_liquidity(r.get("area_sqm"), r.get("ptype"))
    _area_txt = f"{safe_float(r['area_sqm']):.1f}m²"
    if _liq["tier"] == "sub_loan":
        chips.append(("🚪", tr("risk_liquidity"), _area_txt, "risk"))
        details.append(("risk", tr("area_sub_loan", area=f"{safe_float(r['area_sqm']):.1f}",
                                   floor=int(AREA_LOAN_FLOOR),
                                   penalty=int(round(_liq["penalty"])))))
    elif _liq["tier"] == "caution":
        chips.append(("🚪", tr("risk_liquidity"), _area_txt, "caution"))
        details.append(("caution", tr("area_caution", area=f"{safe_float(r['area_sqm']):.1f}",
                                      caution=int(AREA_LENDER_CAUTION),
                                      penalty=int(round(_liq["penalty"])))))
    elif _liq["tier"] == "broad":
        chips.append(("🚪", tr("risk_liquidity"), _area_txt, "ok"))
        details.append(("info", tr("area_broad", owner=int(AREA_OWNER_OCC))))
    elif _liq["tier"] == "investor":
        chips.append(("🚪", tr("risk_liquidity"), _area_txt, "ok"))
        details.append(("info", tr("area_investor", owner=int(AREA_OWNER_OCC))))
    else:
        chips.append(("🚪", tr("risk_liquidity"), tr("rv_unchecked"), "unknown"))

    # Resale outlook from official population projections
    if outlook and outlook.get("band") == "low_base":
        chips.append(("📉", tr("risk_resale"), tr("rv_low_base"), "unknown"))
        details.append(("info", tr("resale_low_base", pop=int(outlook["base_pop"]),
                                   minimum=int(POP_MIN_BASE))))
    elif outlook and outlook.get("band") == "anomaly":
        chips.append(("📉", tr("risk_resale"), tr("rv_anomaly"), "unknown"))
        details.append(("caution", tr("resale_anomaly", pct=f"{outlook['pct_change']:+.1f}",
                                      frm=outlook["from_year"], to=outlook["to_year"],
                                      field=outlook.get("field", "?"))))
    elif outlook:
        band = outlook["band"]
        state = {"severe": "risk", "mild": "caution",
                 "growth": "ok", "flat": "ok"}.get(band, "unknown")
        chips.append(("📈" if band == "growth" else "📉", tr("risk_resale"),
                      f"{outlook['pct_change']:+.1f}% → {outlook['to_year']}", state))
        msg = tr(f"resale_{band}", pct=f"{outlook['pct_change']:+.1f}",
                 frm=outlook["from_year"], to=outlook["to_year"])
        _eff = resale_gate - resale_delta
        if _eff < -0.01:
            msg += " " + tr("resale_penalty_note", extra=f"{abs(_eff):.1f}")
        elif _eff > 0.01:
            msg += " " + tr("resale_credit_note", back=f"{_eff:.1f}")
        details.append((state if state != "ok" else "info", msg))
    else:
        chips.append(("📉", tr("risk_resale"), tr("rv_unchecked"), "unknown"))
        if coords:
            details.append(("info", tr("resale_unknown")))

    # Development context (a positive signal, so its own state)
    if dev_flagged:
        dnames = sep.join(tr(f"dev_{k}") for k in dev_flagged)
        chips.append(("🏗️", tr("risk_development"), f"+{int(dev_bon)}", "plus"))
        details.append(("info", tr("dev_context", items=dnames, bonus=int(dev_bon))))
    elif dev:
        chips.append(("🏗️", tr("risk_development"), tr("rv_none"), "ok"))
    else:
        chips.append(("🏗️", tr("risk_development"), tr("rv_unchecked"), "unknown"))

    st.markdown(risk_strip(chips), unsafe_allow_html=True)
    if details:
        _has_risk = any(sev == "risk" for sev, _ in details)
        with st.expander(tr("risk_detail_header"), expanded=_has_risk):
            for sev, msg in details:
                (st.error if sev == "risk" else
                 st.warning if sev == "caution" else st.info)(msg)

    # --- Verdict + headline metrics ------------------------------------------
    st.markdown(
        f"<div class='qc-card'><div class='qc-ticker'>"
        f"{display_title(r['ptype'], r['prefecture'], r['station_min'], r['building_age'], r['area_sqm'])}"
        f"</div><div style='margin-top:8px;'>{verdict_pill(display_score)}"
        f"&nbsp;&nbsp;<span class='qc-value'>{display_score:.1f} / 100</span></div></div>",
        unsafe_allow_html=True,
    )
    ppsm = price_yen / float(area)
    # Benchmark headline now resolves at the municipality the listing maps to
    # (city_code), falling back to the prefecture inside get_sqm_rate.
    rate, src, n_samples = get_sqm_rate(city_code, pref=pref)
    # Tell the user, up front, exactly which benchmark scored this listing —
    # ward-level when resolved, prefecture average when it fell back — so the
    # hyper-local scoring is visible rather than silent.
    if src == "MLIT_CITY" and city_code:
        st.info(tr("ward_resolved", ward=municipality_label(city_code)))
    elif city_code:
        # The ward WAS identified — we simply have no MLIT data for it yet. Saying
        # "no benchmark for this address" here reads as a lookup failure, which
        # sends you off checking the address when the real fix is a refresh.
        is_target = city_code in MLIT_TARGET_CITIES.get(pref, [])
        st.info(tr("ward_no_data" if is_target else "ward_not_fetched",
                   ward=municipality_label(city_code), pref=pref_name(pref)))
    else:
        st.info(tr("ward_fallback", pref=pref_name(pref)))
    momentum = get_ward_momentum(city_code)
    if momentum:
        mpct, mq, mn = momentum
        st.caption(tr("ward_trend", ward=municipality_label(city_code),
                      pct=mpct, q=mq, n=mn))
    if coords:
        geomock = (_get_secret("GEOCODER", "gsi") or "gsi").lower() == "mock"
        st.caption(tr("geocode_mock" if geomock else "geocode_ok",
                      lat=coords[0], lon=coords[1]))
    elif address:
        st.caption(tr("geocode_miss"))
    delta_pct = (ppsm / rate - 1.0) * 100.0
    src_label = _benchmark_src_label(src, n_samples)
    m1, m2, m3 = st.columns(3)
    # Name the comparison after the municipality when the rate actually came
    # from city-level data; otherwise the prefecture (the honest fallback).
    bench_area = municipality_label(city_code) if src == "MLIT_CITY" else pref_name(pref)
    m1.markdown(metric_card(tr("price_label"), fmt_yen(price_yen)), unsafe_allow_html=True)
    # Below-benchmark ¥/m² is GOOD for a buyer -> green when negative.
    m2.markdown(metric_card(tr("sqm_price_label"), fmt_yen(ppsm),
                            delta=f"{delta_pct:+.1f}% {tr('vs_benchmark', pref=bench_area)}",
                            positive=delta_pct <= 0), unsafe_allow_html=True)
    # Gross is what the listing advertises; net is what the owner keeps. Show
    # gross as the headline with net as the delta so they're never confused.
    # (_ny and _net computed once, immediately after analyze_properties)
    # With no advertised figure the card used to read 0.00%, which looks like a
    # terrible deal rather than a missing input. Show the contractual implied
    # yield instead, and LABEL which number is on screen.
    _adv = safe_float(r.get("gross_yield"))
    _implied = safe_float(_ny.get("contractual_gross_pct"))
    if not math.isnan(_adv) and _adv > 0:
        _y_label, _y_value = tr("yield_label_advertised"), fmt_pct(_adv)
    elif not math.isnan(_implied):
        _y_label, _y_value = tr("yield_label_implied"), fmt_pct(_implied)
    else:
        _y_label, _y_value = tr("yield_label_advertised"), "—"
    m3.markdown(metric_card(
        _y_label, _y_value,
        delta=(tr("net_yield_delta", net=f"{_net:.2f}") if not math.isnan(_net)
               else tr("net_yield_unknown")),
        positive=False), unsafe_allow_html=True)
    st.caption(tr("bench_current", pref=bench_area, rate=rate, src=src_label))
    comp=comparable_valuation(city_code,area,age,row.get("structure"),ptype=row.get("ptype"))
    st.markdown("##### " + ("🏘️ Comparable-property valuation" if get_lang()=="en" else "🏘️ 類似成約事例による評価"))
    if comp["available"]:
        a,b,c=st.columns(3); a.metric("Estimated fair value",fmt_yen(comp["estimate"])); b.metric("Comparable range",f"{fmt_yen(comp['low'])} – {fmt_yen(comp['high'])}"); c.metric("Asking vs fair value",f"{(price_yen/comp['estimate']-1)*100:+.1f}%")
        st.caption(f"{comp['confidence']} confidence · {comp['n']} same-city transactions · median similarity {comp['similarity']:.0%}")
    elif comp.get("unavailable_reason") == "no_commercial_comparables":
        st.info(tr("comp_unavailable_commercial"))
    else: st.info("Refresh the prefecture's MLIT benchmarks to collect comparable transactions; at least five similar records are required.")
    # NOTE: financing is deliberately NOT shown here. Exactly one financing
    # interface exists, in a collapsed expander after the unlevered analysis, so
    # leverage assumptions can never colour the asset-quality read.
    confidence,level=recommendation_confidence(r,src,n_samples,hazards,dev,outlook,address)
    st.markdown("##### 🔎 Recommendation confidence"); st.progress(confidence/100,text=f"{level} · {confidence}%")
    if not math.isnan(_net):
        st.caption(tr("net_yield_breakdown",
                      rent=fmt_yen(_ny["annual_rent"]),
                      vac=f"{_ny['vacancy_applied']:.0f}",
                      basis=yield_basis_name(_ny["yield_basis"]),
                      mgmt=f"{_expense_assumptions()['mgmt_pct']:.0f}",
                      fees=fmt_yen(_ny["annual_fees"]),
                      tax=fmt_yen(_ny["annual_tax"]),
                      noi=fmt_yen(_ny["noi"]), net=f"{_net:.2f}",
                      src=tr("fees_estimated") if _ny["fees_estimated"] else tr("fees_actual")))
        # Yield on capital actually deployed, not just on the sticker price.
        st.caption(tr("invested_yield_note",
                      costs=fmt_yen(_ny["acq_costs"]),
                      pct=f"{acquisition_costs(price_yen)['costs_pct']:.1f}",
                      invested=fmt_yen(_ny["total_invested"]),
                      net_inv=f"{_ny['net_yield_on_invested']:.2f}"))
        if _ny["yield_basis"] == "unknown":
            st.caption(tr("ybasis_warn"))
        # Which rent the base case actually used, and whether the advertised
        # yield agrees with the lease.
        st.caption(tr("rent_basis_note",
                      src=tr(f"rent_source_{_ny['rent_source']}"),
                      c=("—" if math.isnan(safe_float(_ny["contractual_gross_pct"]))
                         else f"{_ny['contractual_gross_pct']:.2f}"),
                      a=("—" if math.isnan(safe_float(_ny["reported_gross_pct"]))
                         else f"{_ny['reported_gross_pct']:.2f}"),
                      gap=("—" if math.isnan(safe_float(_ny["gross_gap_pp"]))
                           else f"{_ny['gross_gap_pp']:+.2f}")))
        if _ny.get("rent_conflict"):
            st.warning(tr("rent_conflict_warn", gap=f"{abs(_ny['gross_gap_pp']):.2f}"))

    # --- Factor breakdown ------------------------------------------------------
    st.markdown(f"##### {tr('deal_factors')}")
    fb = pd.DataFrame({"factor": [factor_name(f) for f in FACTORS],
                       "sub_score": [float(r[f]) for f in FACTORS]})
    chart = (
        alt.Chart(fb)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("sub_score:Q", title=None, scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("factor:N", title=None),
            color=alt.Color("factor:N", legend=None),
            tooltip=["factor", alt.Tooltip("sub_score:Q", format=".1f")],
        )
        .properties(height=160)
    )
    st.altair_chart(chart, width="stretch", theme="streamlit")

    # --- Size resilience: selected production mode + transparent derivation ---
    st.markdown(f"##### {tr('size_header')}")
    st.caption(tr("size_caption"))
    if not _sz["applicable"]:
        st.info(tr("size_na"))
    else:
        zc1, zc2, zc3 = st.columns(3)
        zc1.markdown(metric_card(tr("size_final"), f"{_sz['final']:.0f}"), unsafe_allow_html=True)
        zc2.markdown(metric_card(tr("size_prior_label"), f"{_sz['prior']:.0f}"), unsafe_allow_html=True)
        zc3.markdown(metric_card(tr("size_evidence_label"), f"{_sz['evidence']:.2f}"),
                     unsafe_allow_html=True)
        st.caption(tr("size_breakdown", final=f"{_sz['final']:.1f}", prior=f"{_sz['prior']:.1f}",
                      local=("—" if math.isnan(safe_float(_sz['local'])) else f"{_sz['local']:.1f}"),
                      strength=f"{_sz['evidence']:.2f}", cap=EVIDENCE_STRENGTH_CAP,
                      func=("—" if math.isnan(safe_float(_sz['functional'])) else f"{_sz['functional']:.0f}"),
                      rental=("—" if math.isnan(safe_float(_sz['rental'])) else f"{_sz['rental']:.0f}"),
                      exit=("—" if math.isnan(safe_float(_sz['exit'])) else f"{_sz['exit']:.0f}"),
                      n=_sz["n"], sim=f"{_sz['similarity']:.2f}"))
        if _sz["missing"]:
            st.caption(tr("size_missing", items=", ".join(_sz["missing"][:6])))
        st.caption(tr("size_mode_note", mode=tr(f"sizemode_{get_size_model_mode()}"),
                      adj=f"{safe_float(r.get('size_adjustment'), 0.0):+.1f}", cap=int(SIZE_ADJ_CAP)))
        if not _sz.get("rent_locally_benchmarked", True):
            st.caption(tr("rent_unbenchmarked"))

    # --- Optional financing scenario (second stage, never asset quality) -------
    with st.expander(tr("fin_header")):
        st.caption(tr("fin_caption"))
        g1, g2, g3 = st.columns(3)
        with g1:
            _down = st.slider(tr("fin_down"), 0, 100, step=5, key="man_down_payment")
        with g2:
            _rate = st.number_input(tr("fin_rate"), min_value=0.0, max_value=10.0,
                                    step=0.05, key="man_interest_rate")
        with g3:
            _term = st.slider(tr("fin_term"), 5, 40, step=1, key="man_loan_years")
        _target = st.slider(tr("fin_target_coc"), 0.0, 20.0, step=0.5, key="man_target_coc")
        _noi = compute_net_yield(price_yen, float(r["gross_yield"]), float(r["area_sqm"]),
                                 r.get("ptype"), r.get("monthly_fees_yen"),
                                 r.get("yield_basis") or YIELD_BASIS_DEFAULT,
                            monthly_rent_yen=r.get("monthly_rent_yen"),
                            market_rent_yen=r.get("market_rent_yen"))["noi"]
        fin = financing_analysis(price_yen, _noi, down=_down, rate=_rate,
                                 years=_term, target_coc=_target)
        f1, f2, f3 = st.columns(3)
        f1.markdown(metric_card(tr("fin_status"), tr(f"fin_status_{fin['status']}")),
                    unsafe_allow_html=True)
        f2.markdown(metric_card(tr("fin_dscr"), "—" if math.isinf(fin["dscr"])
                                else f"{fin['dscr']:.2f}"), unsafe_allow_html=True)
        f3.markdown(metric_card(tr("fin_coc"), fmt_pct(fin["coc"])), unsafe_allow_html=True)
        st.caption(tr("fin_breakdown", cash=fmt_yen(fin["cash_invested"]),
                      debt=fmt_yen(fin["debt"]), flow=fmt_yen(fin["cashflow"]),
                      maxp=fmt_yen(fin["max_price"])))
        st.caption(tr("fin_isolation_note"))

    # --- Context vs your own tracked listings ---------------------------------
    # Previously this compared against the latest mock scan, so the percentile
    # was measured against invented listings. Your own ledger is the only honest
    # comparison set — and the one you actually care about.
    _hist = get_historical_picks()
    if not _hist.empty and "score" in _hist.columns:
        _prev = _hist[_hist["listing_id"] != r.get("listing_id")]
        if len(_prev) >= 3:
            pct = float((pd.to_numeric(_prev["score"], errors="coerce")
                         < display_score).mean() * 100.0)
            st.caption(tr("percentile_text", pct=pct, n=len(_prev)))

    if track:
        save_manual_pick(r.to_dict())
        st.success(tr("tracked_success"))

# ----------------------------------------------------------------------------
# Tab 4 — Portfolio Audit (Walk-Forward Tracking)
# ----------------------------------------------------------------------------
STATUS_KEYS = {"sold": "status_sold", "price_drop": "status_price_drop",
               "unchanged": "status_unchanged", "pending": "outcome_pending"}

# What counts as a good call. A listing that sold quickly means the market agreed
# it was well priced; one still sitting after a long time, or discounted, means
# it wasn't. These thresholds are the only judgement in the loop and they are
# stated here rather than buried.
FAST_SALE_DAYS = 30
STALE_DAYS = 120

# Markers portals use when a listing is taken down. Kenbiya shows
# 「広告主により掲載終了となりました」 with the reason (売却済み／販売中止／掲載中止)
# — which tells us the listing is GONE but not reliably WHY, so we record the
# fact and let the ambiguous cases stay unscored.
# HIGH-CONFIDENCE phrases only. An earlier list included bare words like
# 売却済み and 掲載終了, which appear all over portal listing CARDS — so any page
# showing other properties (a homepage, a search page, or the "似た条件の物件"
# block at the bottom of a real listing) matched, and live listings were marked
# sold. These phrases are the full sentence a portal uses about THIS property.
DELISTED_MARKERS = ["広告主により掲載終了", "掲載終了となりました", "掲載を終了しました",
                    "この物件は掲載終了", "この物件の掲載は終了", "販売を終了しました",
                    "ページが見つかりません", "お探しのページは"]
# Everything from here down is other properties, not this one. Recommendation
# blocks are exactly where stray "売却済み" labels live.
RECOMMEND_MARKERS = ["似た条件の物件", "おすすめ物件", "関連物件", "この物件を見た人",
                     "最近見た物件", "似ている物件"]

def _is_real_listing_url(url: str, portal: str | None = None) -> bool:
    """True only for a URL that points at an individual listing PAGE.

    save_manual_pick substitutes the portal homepage when you analyse a listing
    without pasting its URL. Status-checking a homepage is meaningless and was
    actively harmful: homepages are full of listing cards, so they tripped the
    delisting markers and silently marked live properties as sold."""
    try:
        parts = urllib.parse.urlparse(url.strip())
    except Exception:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    path = (parts.path or "").strip("/")
    return len(path) >= 3 and path.count("/") >= 1

def check_listing_status(url: str) -> dict:
    """Check whether ONE tracked listing is still live.

    This is deliberately narrow and worth being precise about: it re-visits a
    single page you already chose to track, to see whether it is still there. It
    does not discover listings, does not walk search results, and does not
    crawl — one request per property you are actively monitoring, which is what
    a person checking their own shortlist does by hand anyway. It reuses the
    same portal allowlist and redirect guard as every other fetch.

    Returns {"state": "delisted"|"live"|"unreachable", "detail": str}.
    NEVER raises.
    """
    try:
        text = _fetch_page_text(url)
    except Exception as e:
        # A portal that blocks server-side fetches (Rakumachi from cloud hosts)
        # lands here. Unreachable is NOT evidence of anything — never treated as
        # a delisting.
        return {"state": "unreachable", "detail": _scrub_secrets(str(e))[:120]}
    flat = unicodedata.normalize("NFKC", _strip_html(text))
    # Cut everything from the first recommendations block onward, so another
    # property's "売却済み" badge can never be read as this one's status.
    cut = min((flat.find(mk) for mk in RECOMMEND_MARKERS if flat.find(mk) > 0),
              default=-1)
    main = flat[:cut] if cut > 0 else flat[:20000]
    for marker in DELISTED_MARKERS:
        if marker in main:
            return {"state": "delisted", "detail": marker}
    return {"state": "live", "detail": ""}

def refresh_tracked_statuses() -> dict:
    """Re-check every tracked listing that isn't already resolved.

    Records outcomes ONLY where the evidence is unambiguous:
      * delisted quickly            -> win   (sold fast is by far the likeliest
                                              reason a listing vanishes early)
      * delisted later              -> recorded as sold, scored NEUTRAL. Per the
                                       product decision that every confirmed
                                       delisting is treated as a sale. Read with
                                       care: 掲載終了 covers 売却済み AND
                                       販売中止/掲載中止, so a withdrawn-unsold
                                       listing is labelled "sold" here. Neutral
                                       scoring keeps that assumption from
                                       flattering the weight learner.
      * still live past STALE_DAYS  -> loss  (unambiguous: it is still sitting there)
      * unreachable                 -> nothing recorded at all
    """
    picks = get_historical_picks()
    out = {"checked": 0, "delisted": 0, "live": 0, "unreachable": 0, "scored": 0,
           "no_url": 0}
    if picks.empty:
        return out
    today = date.today()
    for _, row in picks.iterrows():
        if int(safe_float(row.get("evaluated"), 0) or 0) == 1:
            continue                      # already resolved
        url = str(row.get("url") or "")
        if not _is_real_listing_url(url, row.get("portal")):
            out["no_url"] = out.get("no_url", 0) + 1
            continue        # placeholder/homepage URL: nothing to check
        res = check_listing_status(url)
        out["checked"] += 1
        out[res["state"]] = out.get(res["state"], 0) + 1
        if res["state"] == "unreachable":
            continue
        try:
            rec = date.fromisoformat(str(row.get("recommendation_date"))[:10])
            days = max((today - rec).days, 0)
        except Exception:
            days = 0
        if res["state"] == "delisted":
            record_real_outcome(row["listing_id"], "sold", days)
            if days <= FAST_SALE_DAYS: out["scored"] += 1
        elif days >= STALE_DAYS:
            record_real_outcome(row["listing_id"], "unchanged", days); out["scored"] += 1
        else:
            _set_status_only(row["listing_id"], "unchanged", days)
        time.sleep(MLIT_REQUEST_PAUSE_S)
    _memo_invalidate()
    return out

def clear_outcome(listing_id: str) -> None:
    """Put a listing back to 'pending' — undo a wrong or premature outcome."""
    with get_conn() as conn:
        conn.execute("UPDATE historical_picks SET current_status='pending', "
                     "days_listed=NULL, outcome=NULL, evaluated=0 WHERE listing_id=?",
                     (listing_id,))
    _memo_invalidate()

def _set_status_only(listing_id: str, status: str, days: int) -> None:
    """Record what we observed without claiming it settles the call."""
    with get_conn() as conn:
        conn.execute("UPDATE historical_picks SET current_status=?, days_listed=? "
                     "WHERE listing_id=?", (status, int(days), listing_id))

def record_real_outcome(listing_id: str, status: str, days_listed: int) -> None:
    """Record an OBSERVED outcome for a tracked listing.

    Records outcomes that were actually observed. (An earlier build invented them with a
    random number generator and fed them to the weight learner — so the engine
    was tuning itself on noise. Entered by hand because that is the only honest
    source: you saw the listing sell, get cut, or sit there.

      sold within FAST_SALE_DAYS      -> win  (+1)
      price cut, or still listed past STALE_DAYS -> loss (-1)
      anything else                   -> neutral (0), stays open for revision
    """
    # ORDER MATTERS. A sale is resolved on its own terms before any staleness
    # test: previously the `days_listed >= STALE_DAYS` branch fired regardless of
    # status, so a listing CONFIRMED SOLD at day 150 was recorded as outcome=-1
    # — a failed call, despite having sold. A slow sale is a weak positive, not
    # a loss; only something that did NOT sell can be a loss.
    if status == "sold":
        if days_listed <= FAST_SALE_DAYS:
            outcome, evaluated = 1, 1          # priced to move: a good call
        elif days_listed >= STALE_DAYS:
            outcome, evaluated = 0, 1          # sold, but slowly: no signal either way
        else:
            outcome, evaluated = 0, 1
    elif status == "price_drop" or days_listed >= STALE_DAYS:
        outcome, evaluated = -1, 1             # cut, or still sitting there: a bad call
    else:
        outcome, evaluated = 0, 0              # nothing decisive yet; stays open
    with get_conn() as conn:
        conn.execute(
            "UPDATE historical_picks SET current_status=?, days_listed=?, "
            "outcome=?, evaluated=? WHERE listing_id=?",
            (status, int(days_listed), outcome, evaluated, listing_id))
    _memo_invalidate()

# Fixed allow-list of columns a later-stage edit may touch. Column names are
# interpolated into the UPDATE, so this MUST stay a literal set — never anything
# derived from user input. v17 adds source_url (a URL is not a document title)
# and rent_evidence_summary (what was selected, and how disperse it was).
EDITABLE_PICK_FIELDS = {
    "agency_name", "agent_name", "agency_phone", "agency_address", "agency_role",
    "source_type", "source_document_name", "source_document_date", "source_url",
    "seller_name", "management_company", "rent_guarantee_company",
    "monthly_rent_yen", "market_rent_yen", "occupancy", "provenance",
    "rent_evidence_summary", "local_median_rent_per_sqm",
    "rental_comparable_count", "rent_median_similarity", "rent_recency_months",
    "median_listing_duration_days", "rent_source_kind", "size_resilience",
    "size_evidence", "size_breakdown", "size_adjustment", "size_model_mode", "score",
}

REVISION_KINDS = ("source_update", "agency_update", "lease_update",
                  "market_rent_update", "location_update", "document_update",
                  "manual_correction", "refreshed_analysis")
MAX_REVISION_JSON_BYTES = 64 * 1024   # bound the payload; never store a page

def _revision_payload(obj) -> str | None:
    """JSON for a revision column, size-bounded. Oversized payloads are replaced
    by a marker rather than silently truncated into invalid JSON."""
    if obj in (None, {}, []):
        return None
    blob = json.dumps(obj, ensure_ascii=False, default=str)
    if len(blob.encode("utf-8")) > MAX_REVISION_JSON_BYTES:
        return json.dumps({"_truncated": True,
                           "_bytes": len(blob.encode("utf-8"))}, ensure_ascii=False)
    return blob

def create_property_revision(pick_id: int, kind: str, before: dict, after: dict,
                             *, reason=None, source_name=None, source_url=None,
                             source_document_name=None, ingestion_channel=None,
                             changed_by=None, provenance=None,
                             refreshed_score=None, refreshed_analysis=None) -> dict:
    """Append an immutable revision. Returns {"created", "revision_number", ...}.

    Only CHANGED fields are stored, so a revision says what actually moved. A
    no-op submission creates nothing: an audit trail full of empty revisions is
    harder to read than no trail at all.

    Revision numbers are allocated inside the write and protected by
    UNIQUE(historical_pick_id, revision_number); on collision with a concurrent
    session we retry once rather than dropping the revision.
    """
    if kind not in REVISION_KINDS:
        kind = "manual_correction"
    changed = {k: {"before": before.get(k), "after": v}
               for k, v in (after or {}).items()
               if k in EDITABLE_PICK_FIELDS
               and str(before.get(k) if before.get(k) is not None else "")
               != str(v if v is not None else "")}
    if not changed:
        return {"created": False, "reason": "no_change", "revision_number": None}
    now = datetime.now().isoformat(timespec="seconds")
    for attempt in range(2):
        try:
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT COALESCE(MAX(revision_number), 0) AS n "
                    "FROM property_revisions WHERE historical_pick_id=?",
                    (int(pick_id),)).fetchone()
                nxt = int((row["n"] if row and row["n"] is not None else 0)) + 1
                conn.execute(
                    "INSERT INTO property_revisions (historical_pick_id, revision_number,"
                    " revision_kind, changed_at, changed_by, change_reason, source_name,"
                    " source_url, source_document_name, ingestion_channel, fields_before,"
                    " fields_after, changed_fields, provenance, refreshed_score,"
                    " refreshed_analysis, original_score_preserved)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                    (int(pick_id), nxt, kind, now, changed_by, reason, source_name,
                     source_url, source_document_name, ingestion_channel,
                     _revision_payload({k: v["before"] for k, v in changed.items()}),
                     _revision_payload({k: v["after"] for k, v in changed.items()}),
                     _revision_payload(sorted(changed)),
                     _revision_payload(provenance),
                     _num_or_none(refreshed_score),
                     _revision_payload(refreshed_analysis)))
            _memo_invalidate()
            return {"created": True, "revision_number": nxt, "kind": kind,
                    "changed_fields": sorted(changed)}
        except Exception as exc:
            # A UNIQUE violation means another session took this number. Retry
            # once; never swallow the revision.
            if attempt == 0 and "UNIQUE" in str(exc).upper():
                continue
            return {"created": False, "reason": _scrub_secrets(str(exc))[:140],
                    "revision_number": None}
    return {"created": False, "reason": "collision", "revision_number": None}

def get_property_revisions(pick_id: int) -> list[dict]:
    """All revisions for a property, oldest first."""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM property_revisions WHERE historical_pick_id=? "
                "ORDER BY revision_number", (int(pick_id),)).fetchall()
    except Exception:
        return []
    out = []
    for r in rows:
        d = dict(r)
        for key in ("fields_before", "fields_after", "changed_fields",
                    "provenance", "refreshed_analysis"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except Exception:
                    pass
        out.append(d)
    return out

def get_effective_pick_state(pick_id: int) -> dict:
    """Original recommendation + revisions applied in order.

    ONE place that knows how to fold revisions, so the UI, the analysis and any
    export cannot disagree about what the current state is. The original score
    and recommendation date are returned SEPARATELY and are never overwritten by
    a revision — that distinction is the whole point of the table.
    """
    empty = {"found": False, "effective": {}, "original": {}, "revisions": [],
             "revision_count": 0, "original_score": None,
             "original_recommendation_date": None, "last_revised_at": None}
    try:
        with get_conn() as conn:
            row = conn.execute("SELECT * FROM historical_picks WHERE id=?",
                               (int(pick_id),)).fetchone()
    except Exception:
        return empty
    if not row:
        return empty
    original = dict(row)
    effective = dict(original)
    revisions = get_property_revisions(pick_id)
    for rev in revisions:
        after = rev.get("fields_after") or {}
        if isinstance(after, dict):
            for k, v in after.items():
                if k in EDITABLE_PICK_FIELDS:
                    effective[k] = v
    return {"found": True, "effective": effective, "original": original,
            "revisions": revisions, "revision_count": len(revisions),
            # Preserved separately and deliberately: a refreshed analysis is a
            # different question from "was the original call good?".
            "original_score": original.get("score"),
            "original_recommendation_date": original.get("recommendation_date"),
            "last_revised_at": revisions[-1]["changed_at"] if revisions else None}

def update_pick_details(listing_id: str, updates: dict) -> None:
    clean = {k: v for k, v in updates.items() if k in EDITABLE_PICK_FIELDS}
    if not clean: return
    with get_conn() as conn:
        conn.execute(f"UPDATE historical_picks SET {', '.join(f'{k}=?' for k in clean)} WHERE listing_id=?", tuple(clean.values()) + (listing_id,))
    _memo_invalidate()

# Labels that END a rent context. Anything after one of these is a different
# quantity and must not be read as rent.
_HOMES_STOP_LABELS = ("AI予測価格", "AI査定価格", "予測価格", "査定価格", "参考価格",
                      "販売価格", "売出価格", "成約価格", "価格", "購入", "売買",
                      "管理費", "修繕積立金", "共益費", "敷金", "礼金", "保証金",
                      "仲介手数料", "月額返済", "返済額", "ローン")
# Labels that DISQUALIFY a figure outright, even when a rent word is nearby.
# 月額返済 contains 月額, so a naive rent scan reads a mortgage payment as rent.
_HOMES_EXCLUDE_LABELS = ("月額返済", "返済額", "ローン", "AI予測価格", "AI査定価格",
                         "査定価格", "予測価格", "販売価格", "売出価格", "成約価格",
                         "敷金", "礼金", "保証金", "仲介手数料", "管理費",
                         "修繕積立金", "共益費")
# Rent labels, split by what they actually assert.
_RENT_LABELS_ACTUAL = ("現行賃料", "契約賃料", "現況賃料", "賃料", "家賃", "月額賃料")
_RENT_LABELS_ESTIMATE = ("想定賃料", "参考賃料", "市場賃料", "予想賃料", "想定家賃",
                         "参考家賃", "市場家賃")

def strip_script_blocks(html_or_text: str) -> str:
    """Remove script/style/JSON-LD before any rent parsing.

    Portal pages embed their whole catalogue as JSON inside <script> tags —
    including prices for other properties. Parsing that as visible text invents
    rent observations that no human ever saw on the page."""
    t = html_or_text or ""
    for pat in (r"<script\b[^>]*>.*?</script>", r"<style\b[^>]*>.*?</style>",
                r"<noscript\b[^>]*>.*?</noscript>"):
        t = re.sub(pat, " ", t, flags=re.S | re.I)
    # Bare JSON blobs that survived (application/ld+json already stripped above).
    t = re.sub(r"\{[^{}]*\"@type\"[^{}]*\}", " ", t)
    return t

def _homes_context_window(text: str, label_end: int) -> str:
    """Text belonging to one rent label: to end-of-line, cut at the next
    competing label, so a neighbouring sale price can never be absorbed."""
    ctx = text[label_end:label_end + 90].split("\n", 1)[0]
    cut = min((ctx.find(lbl) for lbl in _HOMES_STOP_LABELS if ctx.find(lbl) > 0),
              default=-1)
    return ctx[:cut] if cut > 0 else ctx

def _yen_in(ctx: str) -> list[float]:
    """Yen figures in a context, refusing to start mid-number.

    「3,027万円」 must never match as 「027万円」 = ¥270,000 — a plausible-looking
    rent conjured out of a ¥30M sale price."""
    # DOCUMENT ORDER matters: a context can hold more than one quote
    # (「賃料 12.5万円 月額 128,000円」) and each becomes its own observation.
    # Scanning 円 and 万円 in separate passes reorders them, and when only the
    # first was taken one of the two was silently dropped.
    found: list[tuple[int, float]] = []
    for m in re.finditer(r"(?<![\d,.])([\d,]{4,9})\s*円", ctx):
        v = float(m.group(1).replace(",", ""))
        if 10_000 <= v <= 1_000_000:
            found.append((m.start(), v))
    for m in re.finditer(r"(?<![\d,.])(\d{1,3}(?:\.\d+)?)\s*万円", ctx):
        v = float(m.group(1)) * 10_000
        if 10_000 <= v <= 1_000_000:
            found.append((m.start(), v))
    out, seen_v = [], set()
    for _, v in sorted(found):
        if v not in seen_v:
            seen_v.add(v)
            out.append(v)
    return out

def extract_rental_observations(page_text: str, source_url: str | None = None,
                                subject_area=None) -> list[dict]:
    """Structured rent candidates for REVIEW — never an automatic market median.

    Returns one record per observation with its raw surrounding text and any
    attributes found beside it (area, layout, floor, management fee), so a human
    can see what each number actually refers to before deciding it is comparable.
    A flat list of numbers cannot express that a figure came from a different
    layout, floor, area or date — which is exactly how a "median" becomes wrong.

    Safeguards enforced here:
      * no rent label, no observation;
      * excluded labels (mortgage payment, deposit, key money, sale/AI price,
        management fee, 共益費) never produce one;
      * rent and management fee are kept SEPARATE unless the page itself labels a
        total;
      * estimates (想定/参考/市場) are marked evidence_kind="estimate", never
        "actual";
      * script/style/JSON-LD is stripped first.
    """
    text = unicodedata.normalize("NFKC", strip_script_blocks(page_text or ""))
    subj_area = safe_float(subject_area)
    observations: list[dict] = []
    seen: set[tuple] = set()
    # Estimate labels are scanned FIRST so the longer 想定賃料 claims its position
    # before the bare 賃料 inside it can. The returned list is re-sorted below so
    # actual figures head the review table — an estimate should never be the
    # first thing a reviewer's eye lands on.
    labels = [(lb, "estimate") for lb in _RENT_LABELS_ESTIMATE] + \
             [(lb, "actual") for lb in _RENT_LABELS_ACTUAL]
    for label, kind in labels:
        for m in re.finditer(re.escape(label), text):
            # A longer label already claimed this position (想定賃料 vs 賃料),
            # or the figure belongs to an excluded quantity.
            lead = text[max(0, m.start() - 6):m.start()]
            if any(x in lead + label for x in _HOMES_EXCLUDE_LABELS):
                continue
            # 想定賃料 CONTAINS 賃料, so the bare actual-rent label also matches
            # inside it and produced a duplicate marked "actual" — an estimate
            # masquerading as an observed rent. If the preceding characters
            # complete an estimate label, this match belongs to that label.
            if kind == "actual" and any((lead + label).endswith(est)
                                        for est in _RENT_LABELS_ESTIMATE):
                continue
            ctx = _homes_context_window(text, m.end())
            rents = _yen_in(ctx)
            if not rents:
                continue
            raw = text[max(0, m.start() - 20):m.start() + 110].replace("\n", " ").strip()
            # Attributes from the SAME line only. A management fee is recorded
            # beside the rent, never added to it: only a figure the page itself
            # labels 総額/合計 may be a total.
            line = text[max(0, text.rfind("\n", 0, m.start()) + 1):
                        (text.find("\n", m.end()) if text.find("\n", m.end()) > 0 else len(text))]
            fee = None
            fm = re.search(r"(?:管理費|共益費)[^\d]{0,6}([\d,]+)\s*円", line)
            if fm:
                fee = float(fm.group(1).replace(",", ""))
            total = None
            tm = re.search(r"(?:総額|合計|総費用)[^\d]{0,6}([\d,]+)\s*円", line)
            if tm:
                total = float(tm.group(1).replace(",", ""))
            am = re.search(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*(?:m2|m²|㎡)", line)
            area = float(am.group(1)) if am else None
            pm = re.search(r"\b([1-9](?:R|K|DK|LDK|SLDK))\b", line)
            fl = re.search(r"(\d{1,2})\s*階", line)
            for rent in rents:
                key = (round(rent), kind)
                if key in seen:
                    continue
                seen.add(key)
                observations.append({
                    "monthly_rent_yen": rent,
                    "management_fee_yen": fee,
                    "total_monthly_cost_yen": total,   # only when the page says so
                    "area_sqm": area,
                    "floor_plan": pm.group(1) if pm else None,
                    "unit_floor": fl.group(1) if fl else None,
                    "evidence_kind": kind,
                    "source_label": label,
                    "raw_context": raw[:240],
                    "source_url": source_url,
                    "listing_status": None,
                    "area_mismatch": (area is not None and not math.isnan(subj_area)
                                      and subj_area > 0
                                      and abs(area - subj_area) / subj_area > 0.20),
                    "selected": False,
                })
    observations.sort(key=lambda o: (o["evidence_kind"] != "actual",
                                     o["monthly_rent_yen"]))
    return observations

def summarise_selected_observations(observations: list[dict]) -> dict:
    """Median and dispersion from the USER-SELECTED observations only.

    Nothing is computed from the full candidate list: a median over unreviewed
    figures is precisely the number this feature exists to avoid producing."""
    chosen = [o for o in (observations or []) if o.get("selected")]
    vals = sorted(float(o["monthly_rent_yen"]) for o in chosen
                  if safe_float(o.get("monthly_rent_yen")) > 0)
    if not vals:
        return {"count": 0, "median": None, "dispersion_pct": None,
                "layouts": [], "selected_ids": [], "mixed_layouts": False}
    med = float(np.median(vals))
    q1, q3 = float(np.quantile(vals, .25)), float(np.quantile(vals, .75))
    layouts = sorted({o.get("floor_plan") for o in chosen if o.get("floor_plan")})
    return {"count": len(vals), "median": med,
            "dispersion_pct": round((q3 - q1) / med * 100, 1) if med else None,
            "layouts": layouts,
            # Mixing layouts in one median is a category error worth flagging.
            "mixed_layouts": len(layouts) > 1,
            "selected_ids": [o.get("observation_id") for o in chosen]}

def build_local_rent_evidence(observations: list[dict], *, subject_area=None,
                              subject_building_age=None, subject_layout=None) -> dict:
    """Convert user-selected rental observations into transparent model inputs.

    Advertised/pasted figures remain ``agent_supported``. Only observations
    explicitly marked achieved or contractual receive achieved-local status.
    Missing attributes are omitted and remaining similarity weights normalize.
    """
    chosen = [o for o in (observations or []) if o.get("selected")]
    sa, sb = safe_float(subject_area), safe_float(subject_building_age)
    sl = str(subject_layout or "").strip().upper()
    ppsm, sims, months, durations = [], [], [], []
    for o in chosen:
        rent, area = safe_float(o.get("monthly_rent_yen")), safe_float(o.get("area_sqm"))
        if not math.isnan(rent) and rent > 0 and not math.isnan(area) and area > 0:
            ppsm.append(rent / area)
        vals, weights = [], []
        if not math.isnan(area) and area > 0 and not math.isnan(sa) and sa > 0:
            vals.append(max(0.0, 1.0 - abs(area - sa) / sa)); weights.append(0.60)
        age = safe_float(o.get("building_age"))
        if not math.isnan(age) and not math.isnan(sb):
            vals.append(max(0.0, 1.0 - abs(age - sb) / 30.0)); weights.append(0.25)
        layout = str(o.get("floor_plan") or "").strip().upper()
        if layout and sl:
            vals.append(1.0 if layout == sl else 0.35); weights.append(0.15)
        if weights:
            sims.append(sum(v*w for v,w in zip(vals,weights)) / sum(weights))
        observed = o.get("observed_at")
        if observed:
            try:
                d = datetime.fromisoformat(str(observed).replace("Z", "+00:00")).date()
                months.append(max((date.today() - d).days / 30.4375, 0.0))
            except Exception:
                pass
        days = safe_float(o.get("listing_days"))
        if not math.isnan(days) and days >= 0:
            durations.append(days)
    kinds = {str(o.get("evidence_kind") or "").lower() for o in chosen}
    source = "achieved_local" if kinds and kinds <= {"achieved", "contractual"} else "agent_supported"
    return {
        "local_median_rent_per_sqm": float(np.median(ppsm)) if ppsm else None,
        "rental_comparable_count": len(chosen),
        "rent_median_similarity": float(np.median(sims)) if sims else None,
        "rent_recency_months": float(np.median(months)) if months else None,
        "median_listing_duration_days": float(np.median(durations)) if durations else None,
        "rent_source_kind": source if chosen else None,
    }


def recompute_pick_size_from_rent_evidence(row, evidence: dict) -> dict:
    """Refresh only the bounded size component after reviewed rent evidence."""
    merged = {**dict(row), **evidence}
    res = size_resilience(merged.get("area_sqm"), merged.get("ptype"),
                          city_code=merged.get("city_code"),
                          building_age=merged.get("building_age"),
                          structure=merged.get("structure"),
                          rent_evidence=_rent_evidence_from_row(merged))
    mode = str(merged.get("size_model_mode") or get_size_model_mode())
    old_adj = safe_float(merged.get("size_adjustment"), 0.0)
    new_adj = (round(size_score_adjustment(res), 2) if mode == "evidence_weighted"
               else 0.0 if mode == "descriptive_only" else old_adj)
    score = max(0.0, min(100.0, safe_float(merged.get("score"), 0.0) - old_adj + new_adj))
    return {**evidence,
            "size_resilience": res.get("final") if res.get("applicable") else None,
            "size_evidence": res.get("evidence"),
            "size_breakdown": json.dumps(res, ensure_ascii=False, default=str),
            "size_adjustment": new_adj, "size_model_mode": mode,
            "score": round(score, 1)}


def save_rental_observations(observations: list[dict], pick_id=None) -> int:
    """Persist reviewed observations. Best-effort; never raises into the UI."""
    if not observations:
        return 0
    now = datetime.now().isoformat(timespec="seconds")
    cols = ["historical_pick_id", "source_url", "observed_at", "monthly_rent_yen",
            "management_fee_yen", "total_monthly_cost_yen", "area_sqm",
            "floor_plan", "unit_floor", "building_age", "listing_status",
            "evidence_kind", "source_label", "raw_context", "selected",
            "confidence", "created_at"]
    n = 0
    try:
        with get_conn() as conn:
            for o in observations:
                conn.execute(
                    f"INSERT INTO rental_observations ({','.join(cols)}) "
                    f"VALUES ({','.join(['?'] * len(cols))})",
                    (pick_id, o.get("source_url"), o.get("observed_at") or now,
                     _num_or_none(o.get("monthly_rent_yen")),
                     _num_or_none(o.get("management_fee_yen")),
                     _num_or_none(o.get("total_monthly_cost_yen")),
                     _num_or_none(o.get("area_sqm")), o.get("floor_plan"),
                     o.get("unit_floor"), _num_or_none(o.get("building_age")),
                     o.get("listing_status"), o.get("evidence_kind"),
                     o.get("source_label"), o.get("raw_context"),
                     1 if o.get("selected") else 0,
                     _num_or_none(o.get("confidence")), now))
                n += 1
    except Exception:
        return n
    return n

def render_listing_deep_dive(picks: pd.DataFrame) -> None:
    """Everything recorded about one tracked listing, in one place.

    The ledger table answers "what am I tracking"; this answers "what did I
    actually know about this one, and what has the ward done since". Useful in
    particular once a listing delists — that is the moment you want the full
    record, not a row in a table."""
    st.markdown(f"##### {tr('deep_header')}")
    labels = {row["listing_id"]: listing_label(row) for _, row in picks.iterrows()}
    sel = st.selectbox(tr("deep_pick"), list(labels),
                       format_func=lambda i: labels.get(i, i), key="deep_pick")
    row = picks[picks["listing_id"] == sel].iloc[0]

    with st.expander(tr("edit_header")):
        suffix = re.sub(r"[^A-Za-z0-9_]", "_", str(sel))[-40:]
        with st.form(f"edit_pick_{suffix}"):
            e1, e2 = st.columns(2)
            with e1:
                agency_edit = st.text_input(tr("edit_agency"), value=str(row.get("agency_name") or ""), key=f"edit_agency_{suffix}")
                agent_edit = st.text_input(tr("edit_agent"), value=str(row.get("agent_name") or ""), key=f"edit_agent_{suffix}")
                source_edit = st.text_input(tr("edit_source"), value=str(row.get("source_document_name") or ""), key=f"edit_source_{suffix}")
            with e2:
                manager_edit = st.text_input(tr("edit_manager"), value=str(row.get("management_company") or ""), key=f"edit_manager_{suffix}")
                contract_edit = st.number_input(tr("edit_contract"), min_value=0, value=int(safe_float(row.get("monthly_rent_yen"), 0.0)), step=1000, key=f"edit_contract_{suffix}")
                market_edit = st.number_input(tr("edit_market"), min_value=0, value=int(safe_float(row.get("market_rent_yen"), 0.0)), step=1000, key=f"edit_market_{suffix}")
            st.text_input(tr("edit_reason"), key=f"edit_reason_{suffix}",
                          placeholder=tr("edit_reason_ph"))
            save_edit = st.form_submit_button(tr("edit_save"), width="stretch")
        if save_edit:
            try: prov = json.loads(row.get("provenance") or "{}")
            except Exception: prov = {}
            prov["later_stage_update"] = {"method": "manual_override", "saved_at": datetime.now().isoformat(timespec="seconds")}
            _pid = int(row.get("id")) if row.get("id") is not None else None
            _submitted = {
                "agency_name": agency_edit.strip() or None,
                "agent_name": agent_edit.strip() or None,
                "source_document_name": source_edit.strip() or None,
                "management_company": manager_edit.strip() or None,
                "monthly_rent_yen": float(contract_edit) if contract_edit else None,
                "market_rent_yen": float(market_edit) if market_edit else None,
            }
            # Append an immutable revision FIRST, then refresh the convenience
            # projection. The revision is the record of what changed and why;
            # the projection just makes the current state cheap to read.
            _rev = {"created": False}
            if _pid is not None:
                _before = get_effective_pick_state(_pid)["effective"]
                _rev = create_property_revision(
                    _pid, "manual_correction", _before, _submitted,
                    reason=(st.session_state.get(f"edit_reason_{suffix}") or None),
                    ingestion_channel="manual", provenance=prov)
            update_pick_details(sel, {**_submitted,
                                      "provenance": json.dumps(prov, ensure_ascii=False)})
            if _rev.get("created"):
                st.caption(tr("rev_created", n=_rev["revision_number"],
                              fields=", ".join(_rev["changed_fields"])))
            elif _rev.get("reason") == "no_change":
                st.caption(tr("rev_no_change"))
            st.success(tr("edit_saved")); st.rerun()
        st.markdown(f"###### {tr('homes_header')}")
        st.caption(tr("homes_caption"))
        evidence_source = st.text_input(tr("homes_url"), key=f"rent_source_{suffix}")
        evidence_text = st.text_area(tr("rent_paste_label"), key=f"rent_text_{suffix}",
                                     height=160, help=tr("rent_paste_help"))
        if st.button(tr("homes_scan"), key=f"rent_parse_{suffix}", width="stretch"):
            if not evidence_text.strip():
                st.info(tr("homes_only"))
            else:
                obs = extract_rental_observations(
                    evidence_text, source_url=evidence_source.strip() or None,
                    subject_area=row.get("area_sqm"))
                now = datetime.now().isoformat(timespec="seconds")
                for o in obs:
                    o["observed_at"] = now
                st.session_state[f"rent_obs_{suffix}"] = obs
        obs = st.session_state.get(f"rent_obs_{suffix}") or []
        if obs:
            st.caption(tr("homes_select_prompt", n=len(obs)))
            for i, o in enumerate(obs):
                cols = st.columns([1, 3, 6])
                with cols[0]:
                    o["selected"] = st.checkbox("", value=bool(o.get("selected")),
                        key=f"rent_sel_{suffix}_{i}", label_visibility="collapsed")
                with cols[1]:
                    bits = [fmt_yen(o["monthly_rent_yen"])]
                    if o.get("floor_plan"): bits.append(str(o["floor_plan"]))
                    if o.get("area_sqm"): bits.append(f"{o['area_sqm']}m²")
                    st.write(" · ".join(bits))
                    tags = [tr(f"evkind_{o['evidence_kind']}")]
                    if o.get("area_mismatch"): tags.append(tr("homes_area_mismatch"))
                    st.caption(" · ".join(tags))
                with cols[2]:
                    st.caption(f"「{o.get('raw_context','')[:110]}」")
            summary = summarise_selected_observations(obs)
            if summary["count"] == 0:
                st.info(tr("homes_none_selected"))
            else:
                st.caption(tr("homes_summary", n=summary["count"],
                              median=fmt_yen(summary["median"]),
                              disp=(f"{summary['dispersion_pct']:.0f}"
                                    if summary["dispersion_pct"] is not None else "—")))
                if summary["mixed_layouts"]:
                    st.warning(tr("homes_mixed_layouts", layouts=", ".join(summary["layouts"])))
                if st.button(tr("homes_use_median"), key=f"rent_use_{suffix}", width="stretch"):
                    try: prov = json.loads(row.get("provenance") or "{}")
                    except Exception: prov = {}
                    model_ev = build_local_rent_evidence(
                        obs, subject_area=row.get("area_sqm"),
                        subject_building_age=row.get("building_age"))
                    updates = recompute_pick_size_from_rent_evidence(row, model_ev)
                    prov["market_rent_yen"] = {
                        "source": evidence_source.strip() or "manual pasted evidence",
                        "change_method": "reviewed_manual_evidence",
                        "selected_count": summary["count"],
                        "candidate_count": len(obs),
                        "dispersion_pct": summary["dispersion_pct"],
                        "changed_at": datetime.now().isoformat(timespec="seconds")}
                    save_rental_observations(obs, pick_id=row.get("id"))
                    updates.update({"market_rent_yen": summary["median"],
                                    "source_url": evidence_source.strip() or None,
                                    "rent_evidence_summary": json.dumps(summary, ensure_ascii=False, default=str),
                                    "provenance": json.dumps(prov, ensure_ascii=False)})
                    update_pick_details(sel, updates)
                    st.success(tr("homes_saved")); st.rerun()
        elif evidence_text.strip() and st.session_state.get(f"rent_obs_{suffix}") == []:
            st.info(tr("homes_no_observations"))
    # --- headline -----------------------------------------------------------
    st.markdown(verdict_pill(safe_float(row.get("score"), 0.0)), unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(metric_card(tr("col_price"), fmt_yen(safe_float(row.get("price_yen")))),
                unsafe_allow_html=True)
    d2.markdown(metric_card(tr("col_area"), f"{safe_float(row.get('area_sqm')):.1f} m²"),
                unsafe_allow_html=True)
    _g, _n = safe_float(row.get("gross_yield")), safe_float(row.get("net_yield"))
    d3.markdown(metric_card(tr("yield_label"), fmt_pct(_g),
                            delta=(tr("net_yield_delta", net=f"{_n:.2f}")
                                   if not math.isnan(_n) else None), positive=False),
                unsafe_allow_html=True)
    d4.markdown(metric_card(tr("col_score"), f"{safe_float(row.get('score')):.1f}"),
                unsafe_allow_html=True)

    # --- the risk picture as recorded at the time ---------------------------
    # Stored flags are compact codes ("flood:rank2", "road,chiku"); they get
    # decoded here. Where a field predates the column that holds it, we RECOMPUTE
    # from the raw inputs we did record rather than showing "not checked", which
    # would wrongly imply the check was never possible.
    chips: list[tuple[str, str, str, str]] = []

    hz_txt = hazard_flags_label(row.get("hazard_flags"))
    hz_short = hazard_flags_label(row.get("hazard_flags"), short=True)
    chips.append(("🌊", tr("risk_hazard"), hz_short or tr("rv_clear"),
                  "risk" if hz_txt else "ok"))

    _s = str(row.get("seismic") or "").strip().lower()
    if _s not in ("old", "grey", "new"):
        _s, _yr = seismic_class(row.get("building_age"))     # recompute from age
    else:
        _yr = seismic_class(row.get("building_age"))[1]
    chips.append(("🏚️", tr("risk_seismic"),
                  {"old": tr("rv_old_seismic", year=_yr or "?"),
                   "grey": tr("rv_grey_seismic", year=_yr or "?"),
                   "new": tr("rv_new_seismic", year=_yr or "?")}.get(_s, tr("rv_unchecked")),
                  {"old": "risk", "grey": "caution", "new": "ok"}.get(_s, "unknown")))

    _at = str(row.get("area_tier") or "").strip().lower()
    if _at not in ("sub_loan", "caution", "investor", "broad"):
        _at = area_liquidity(row.get("area_sqm"), row.get("ptype"))["tier"]  # recompute
    _area = safe_float(row.get("area_sqm"))
    chips.append(("🚪", tr("risk_liquidity"),
                  f"{area_tier_name(_at)} · {_area:.1f}m²" if not math.isnan(_area)
                  else area_tier_name(_at),
                  {"sub_loan": "risk", "caution": "caution",
                   "investor": "ok", "broad": "ok"}.get(_at, "unknown")))

    _po = safe_float(row.get("pop_outlook_pct"))
    chips.append(("📉" if (math.isnan(_po) or _po < 0) else "📈", tr("risk_resale"),
                  f"{_po:+.1f}% " + tr("rv_to_horizon") if not math.isnan(_po)
                  else tr("rv_unchecked"),
                  "unknown" if math.isnan(_po) else
                  "risk" if _po <= RESALE_DECLINE_SEVERE else
                  "caution" if _po <= RESALE_DECLINE_MILD else "ok"))

    dv_txt = dev_flags_label(row.get("dev_flags"))
    dv_short = dev_flags_label(row.get("dev_flags"), short=True)
    chips.append(("🏗️", tr("risk_development"), dv_short or tr("rv_none"),
                  "plus" if dv_txt else "ok"))

    st.markdown(risk_strip(chips), unsafe_allow_html=True)
    # The chips are a summary; this spells out what each one actually means for
    # the deal, which is the part that is hard to infer from a short label.
    with st.expander(tr("deep_risk_explain")):
        if hz_txt:
            st.error(tr("deep_hz_line", zones=hz_txt))
        else:
            st.success(tr("deep_hz_none"))
        if _s == "old":
            st.error(tr("seismic_old", year=_yr or "?",
                        penalty=int(round(get_old_seismic_penalty()))))
        elif _s == "grey":
            st.warning(tr("seismic_grey", year=_yr or "?"))
        elif _s == "new":
            st.success(tr("deep_seismic_ok", year=_yr or "?"))
        if _at == "sub_loan":
            st.error(tr("area_sub_loan", area=f"{_area:.1f}", floor=int(AREA_LOAN_FLOOR),
                        penalty=int(round(get_small_unit_penalty()))))
        elif _at == "caution":
            st.warning(tr("area_caution", area=f"{_area:.1f}",
                          caution=int(AREA_LENDER_CAUTION),
                          penalty=int(round(get_small_unit_penalty() * 0.5))))
        elif _at == "broad":
            st.info(tr("area_broad", owner=int(AREA_OWNER_OCC)))
        elif _at == "investor":
            st.info(tr("area_investor", owner=int(AREA_OWNER_OCC)))
        if not math.isnan(_po):
            band = ("severe" if _po <= RESALE_DECLINE_SEVERE else
                    "mild" if _po <= RESALE_DECLINE_MILD else
                    "growth" if _po >= RESALE_GROWTH else "flat")
            st.info(tr(f"resale_{band}", pct=f"{_po:+.1f}", frm="—", to="—"))
        else:
            st.info(tr("deep_resale_none"))
        if dv_txt:
            st.info(tr("deep_dev_line", items=dv_txt))
        else:
            st.info(tr("deep_dev_none"))

    # --- ward price history, quarter by quarter -----------------------------
    cc = str(row.get("city_code") or "")
    if cc:
        with get_conn() as conn:
            if isinstance(conn, _PgConnWrapper):
                q = pd.DataFrame(conn.execute("SELECT year, quarter, median_ppsm, sample_count FROM municipality_quarterly WHERE city_code=? ORDER BY year, quarter", (cc,)).fetchall())
            else:
                q = pd.read_sql_query("SELECT year, quarter, median_ppsm, sample_count FROM municipality_quarterly WHERE city_code=? ORDER BY year, quarter", conn, params=(cc,))
        if not q.empty:
            q["period"] = q["year"].astype(int).astype(str) + " Q" + q["quarter"].astype(int).astype(str)
            st.markdown(f"###### {tr('deep_ward_chart', ward=municipality_label(cc))}")
            chart = (alt.Chart(q).mark_bar()
                     .encode(x=alt.X("period:N", title=None, sort=list(q["period"])),
                             y=alt.Y("median_ppsm:Q", title=tr("col_sqm_price")),
                             tooltip=[alt.Tooltip("period:N", title=""),
                                      alt.Tooltip("median_ppsm:Q", format=",.0f",
                                                  title=tr("col_sqm_price")),
                                      alt.Tooltip("sample_count:Q", title=tr("col_samples"))])
                     .properties(height=200))
            st.altair_chart(chart, width="stretch")
            mom = get_ward_momentum(cc)
            if mom:
                st.caption(tr("ward_trend", ward=municipality_label(cc),
                              pct=mom[0], q=mom[1], n=mom[2]))
        else:
            st.caption(tr("deep_no_ward_data", ward=municipality_label(cc)))

    # --- Original recommendation vs current revised state ---------------------
    # Shown side by side and clearly labelled. The original score is what the
    # walk-forward learner judges; a refreshed view answers a different
    # question ("what would I conclude today?") and must not be mistaken for it.
    _pid_dd = int(row.get("id")) if row.get("id") is not None else None
    if _pid_dd is not None:
        _state = get_effective_pick_state(_pid_dd)
        if _state["found"] and _state["revision_count"]:
            st.markdown(f"##### {tr('rev_header')}")
            oc1, oc2 = st.columns(2)
            with oc1:
                st.markdown(f"**{tr('rev_original')}**")
                st.caption(tr("rev_original_note",
                              score=f"{safe_float(_state['original_score']):.1f}",
                              date=_state["original_recommendation_date"]))
            with oc2:
                st.markdown(f"**{tr('rev_current')}**")
                st.caption(tr("rev_current_note", n=_state["revision_count"],
                              at=_state["last_revised_at"] or "—"))
            st.caption(tr("rev_score_not_rewritten"))
            with st.expander(tr("rev_history", n=_state["revision_count"])):
                for rv in _state["revisions"]:
                    before = rv.get("fields_before") or {}
                    after = rv.get("fields_after") or {}
                    st.markdown(
                        f"**#{rv['revision_number']}** · {html.escape(str(rv['revision_kind']))}"
                        f" · {html.escape(str(rv['changed_at']))}")
                    for fld in (rv.get("changed_fields") or []):
                        # Revision values are user/document supplied: escape.
                        st.write(f"• {html.escape(str(fld))}: "
                                 f"{html.escape(str(before.get(fld)))} → "
                                 f"{html.escape(str(after.get(fld)))}")
                    if rv.get("change_reason"):
                        st.caption(html.escape(str(rv["change_reason"])))
                    if rv.get("source_name") or rv.get("source_url"):
                        st.caption(html.escape(str(rv.get("source_name")
                                                   or rv.get("source_url"))))

    # --- the full record ----------------------------------------------------
    with st.expander(tr("deep_all_fields")):
        skip = {"id", "factor_snapshot"}
        # Cast to str: this column mixes floats, ints, JSON blobs and None, and
        # Arrow cannot serialise a mixed-type object column without complaining.
        # Also renders a null as an empty cell rather than the literal "nan".
        recs = [{"field": k, "value": "" if row[k] is None else str(row[k])}
                for k in row.index
                if k not in skip and str(row[k]) not in ("None", "nan", "")]
        st.dataframe(pd.DataFrame(recs), width="stretch", hide_index=True)

def render_portfolio_audit() -> None:
    st.markdown(f"### {tr('audit_header')}")
    st.caption(tr("audit_intro"))

    picks = get_historical_picks()
    if picks.empty:
        st.info(tr("audit_empty"))
        return

    # --- Automated status check ---------------------------------------------
    # Re-visits each tracked listing's own page to see if it is still live. One
    # request per property you are already monitoring — not discovery, not a
    # crawl. Portals that block server-side fetches simply report unreachable,
    # which is never treated as evidence of anything.
    if st.button(tr("statuscheck_btn"), width="stretch"):
        with st.spinner(tr("statuscheck_spinner")):
            res = refresh_tracked_statuses()
        if res["checked"] == 0:
            st.info(tr("statuscheck_none"))
        else:
            st.success(tr("statuscheck_done", checked=res["checked"],
                          delisted=res["delisted"], live=res["live"],
                          scored=res["scored"]))
            if res["unreachable"]:
                st.warning(tr("statuscheck_unreachable", n=res["unreachable"]))
        if res.get("no_url"):
            st.info(tr("statuscheck_no_url", n=res["no_url"]))
        st.rerun()

    # --- Record what actually happened -------------------------------------
    # Outcomes are entered by hand, on purpose. The previous version simulated
    # them with a random number generator and fed the results into the weight
    # learner, which meant the "engine learning" was learning from dice. Real
    # outcomes are the only ones worth learning from, even if they arrive slowly.
    with st.expander(tr("outcome_header")):
        st.caption(tr("outcome_help"))
        # Every tracked listing, not just the pending ones — an outcome recorded
        # by mistake has to be reachable in order to be corrected.
        target = picks
        labels = {row["listing_id"]: listing_label(row) for _, row in target.iterrows()}
        opts = list(labels)
        if opts:
            oc1, oc2 = st.columns([2, 1])
            with oc1:
                pick_id = st.selectbox(tr("outcome_listing"), opts,
                                       format_func=lambda i: labels.get(i, i),
                                       key="outcome_pick")
            with oc2:
                new_status = st.selectbox(tr("outcome_status"),
                                          ["sold", "price_drop", "unchanged"],
                                          format_func=lambda k: tr(STATUS_KEYS[k]),
                                          key="outcome_status")
            days = st.number_input(tr("outcome_days"), min_value=0, max_value=2000,
                                   step=1, key="outcome_days")
            b1, b2 = st.columns(2)
            with b1:
                if st.button(tr("outcome_save"), width="stretch"):
                    record_real_outcome(pick_id, new_status, int(days))
                    st.success(tr("outcome_saved", listing=labels.get(pick_id, pick_id)))
                    st.rerun()
            with b2:
                if st.button(tr("outcome_clear"), width="stretch"):
                    clear_outcome(pick_id)
                    st.success(tr("outcome_cleared", listing=labels.get(pick_id, pick_id)))
                    st.rerun()

    # --- Headline accuracy cards -------------------------------------------
    # "Checked" = the reality check has visited the pick at least once (status no
    # longer 'pending'); neutral picks remain evaluated=0 so they re-enter future
    # checks, but they HAVE been looked at and should count as checked.
    checked = picks[picks["current_status"] != "pending"]
    decisive = picks[picks["outcome"].isin([1, -1]) & (picks["evaluated"] == 1)]
    wins = int((decisive["outcome"] == 1).sum())
    winrate = (wins / len(decisive) * 100) if len(decisive) else float("nan")
    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card(tr("checked_label"), f"{len(checked)} / {len(picks)}"), unsafe_allow_html=True)
    c2.markdown(metric_card(tr("accuracy_label"), f"{wins} / {len(decisive) or 0}"), unsafe_allow_html=True)
    c3.markdown(metric_card(tr("winrate_label"), fmt_pct(winrate) if not math.isnan(winrate) else "—",
                            delta=None), unsafe_allow_html=True)

    # --- Original price vs current status ledger ----------------------------
    st.markdown(f"##### {tr('audit_table_header')}")
    ledger = picks.copy()
    ledger["status_disp"] = ledger.apply(
        lambda r: tr(STATUS_KEYS.get(r["current_status"], "outcome_pending")), axis=1)
    if (picks.get("manual", pd.Series(dtype=int)).fillna(0) == 1).any():
        st.caption(tr("manual_audit_note"))
    def _outcome_disp(r):
        if r.get("manual") == 1 and r["current_status"] == "pending":
            return tr("outcome_manual")
        if r["current_status"] == "pending": return tr("outcome_pending")
        if r["outcome"] == 1: return tr("outcome_win")
        if r["outcome"] == -1: return tr("outcome_loss")
        return tr("outcome_neutral")   # checked, but no decisive verdict yet
    ledger["outcome_disp"] = ledger.apply(_outcome_disp, axis=1)
    # Rebuild localized titles; rows from a pre-ptype database fall back to the
    # canonical stored title rather than erroring.
    def _ledger_title(r):
        # Rows written before the ptype/area_sqm migrations fall back to the
        # canonical stored title rather than erroring. The area guard is > 0
        # (not just notna): a NULL became NaN here, but a 0.0 — possible from
        # a hand-edited DB or a defaulted column — would otherwise render an
        # ugly "… · 0 m² · …" title.
        area = safe_float(r.get("area_sqm"), 0.0)
        if pd.notna(r.get("ptype")) and r.get("ptype") and area > 0:
            return display_title(r["ptype"], r["prefecture"], r["station_min"],
                                 r["building_age"], area)
        return r["title"]
    ledger["building_display"] = ledger.apply(listing_label, axis=1)
    ledger["title"] = ledger.apply(_ledger_title, axis=1)
    ledger["unit_display"] = ledger.apply(
        lambda x: " / ".join(v for v in [str(x.get("unit_number") or "").strip(),
                                           ((str(x.get("unit_floor") or "").strip() + "F")
                                            if str(x.get("unit_floor") or "").strip() else "")]
                             if v and v.lower() != "nan"), axis=1)
    ledger["source_display"] = ledger.apply(source_display, axis=1)
    ledger["prefecture"] = ledger["prefecture"].map(pref_name)

    disp = ledger[["recommendation_date", "building_display", "unit_display", "title", "prefecture", "area_sqm", "price_yen",
                   "current_price", "status_disp", "days_listed", "score",
                   "outcome_disp", "source_display", "url"]].rename(columns={
        "recommendation_date": tr("col_rec_date"), "building_display": tr("col_building"),
        "unit_display": tr("col_unit_floor"), "title": tr("col_title"),
        "prefecture": tr("col_prefecture"), "area_sqm": tr("col_area"),
        "price_yen": tr("col_orig_price"),
        "current_price": tr("col_price"), "status_disp": tr("col_current_status"),
        "days_listed": tr("col_days_listed"), "score": tr("col_score"),
        "outcome_disp": tr("col_outcome"), "source_display": tr("col_source"),
        "url": tr("col_url"),
    })
    st.dataframe(
        disp, width="stretch", hide_index=True,
        column_config={
            tr("col_url"): st.column_config.LinkColumn(tr("col_url"), display_text="↗"),
            tr("col_orig_price"): st.column_config.NumberColumn(tr("col_orig_price"), format="¥%.0f"),
            tr("col_price"): st.column_config.NumberColumn(tr("col_price"), format="¥%.0f"),
            tr("col_area"): st.column_config.NumberColumn(tr("col_area"), format="%.1f"),
            tr("col_score"): st.column_config.NumberColumn(tr("col_score"), format="%.1f"),
        },
    )

    # --- The AI learning: live weights + evolution chart ---------------------
    st.markdown(f"##### {tr('learning_header')}")
    st.caption(tr("learning_caption", fast=FAST_SALE_DAYS, stale=STALE_DAYS, floor=MIN_WEIGHT))

    st.divider()
    render_listing_deep_dive(picks)
    weights = get_latest_weights()
    wcols = st.columns(len(FACTORS))
    for col, f in zip(wcols, FACTORS):
        delta = weights[f] - DEFAULT_WEIGHTS[f]
        col.markdown(
            metric_card(factor_name(f), f"{weights[f]:.1%}",
                        delta=f"{delta:+.1%}", positive=delta >= 0),
            unsafe_allow_html=True,
        )

    hist = get_weight_history()
    if len(hist) > len(FACTORS):   # more than just the initial seed row-set
        st.markdown(f"##### {tr('weight_history_header')}")
        hist = hist.assign(factor_label=hist["factor_name"].map(factor_name))
        chart = (
            alt.Chart(hist)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y("current_weight:Q", title=tr("col_weight"), axis=alt.Axis(format=".0%")),
                color=alt.Color("factor_label:N", title=tr("col_factor")),
                tooltip=[alt.Tooltip("timestamp:T"), "factor_label",
                         alt.Tooltip("current_weight:Q", format=".1%")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch", theme="streamlit")

# ----------------------------------------------------------------------------
# Tab 4 — KPI Weights Customizer
# ----------------------------------------------------------------------------
def render_weights_customizer() -> None:
    st.markdown(f"### {tr('weights_header')}")
    # --- Location gates: tunable, session-scoped (defaults are the constants).
    st.markdown(f"##### {tr('gates_header')}")
    st.caption(tr("gates_caption"))
    g1, g2 = st.columns(2)
    with g1:
        st.slider(tr("hazard_pen_label"), 0, 30, int(HAZARD_PENALTY), 1, key="hazard_penalty")
        st.slider(tr("seismic_pen_label"), 0, 30, int(OLD_SEISMIC_PENALTY), 1, key="seismic_penalty")
        st.slider(tr("small_unit_pen_label"), 0, 25, int(SMALL_UNIT_PENALTY), 1, key="small_unit_penalty")
        st.slider(tr("resale_horizon_label"), 5, 30, int(RESALE_HORIZON_YEARS), 5, key="resale_horizon")

    st.markdown(f"##### {tr('verdict_thresh_header')}")
    st.caption(tr("verdict_thresh_caption"))
    v1, v2 = st.columns(2)
    with v1:
        st.slider(tr("verdict_consider_label"), 30, 80, int(VERDICT_CONSIDER), 1,
                  key="verdict_consider")
    with v2:
        st.slider(tr("verdict_strong_label"), 40, 95, int(VERDICT_STRONG), 1,
                  key="verdict_strong")
    with g2:
        st.slider(tr("dev_bonus_label"), 0, 10, int(DEV_BONUS), 1, key="dev_bonus")

    # --- Net-yield expense assumptions: every input to the NOI model, visible.
    st.markdown(f"##### {tr('expenses_header')}")
    st.caption(tr("expenses_caption"))
    e1, e2 = st.columns(2)
    with e1:
        st.slider(tr("exp_fee_label"), 0, 800, int(FEE_PER_SQM_MONTH), 25, key="exp_fee_per_sqm")
        st.slider(tr("exp_vacancy_label"), 0.0, 25.0, float(VACANCY_PCT), 0.5, key="exp_vacancy_pct")
        st.slider(tr("exp_tax_label"), 0.0, 2.0, float(TAX_PCT_OF_PRICE), 0.05, key="exp_tax_pct")
    with e2:
        st.slider(tr("exp_owner_label"), 0.0, 30.0, float(OWNER_MAINT_PCT), 0.5, key="exp_owner_maint_pct")
        st.slider(tr("exp_mgmt_label"), 0.0, 15.0, float(MGMT_PCT), 0.5, key="exp_mgmt_pct")
        st.slider(tr("exp_acq_label"), 0.0, 8.0, float(ACQ_OTHER_PCT), 0.25, key="exp_acq_other_pct")
    st.caption(tr("weights_intro"))

    weights = get_latest_weights()
    sliders = {}
    cols = st.columns(len(FACTORS))
    for col, f in zip(cols, FACTORS):
        with col:
            sliders[f] = st.slider(
                factor_name(f), min_value=float(MIN_WEIGHT), max_value=0.90,
                value=float(round(weights[f], 2)), step=0.01,
                key=f"wslider_{f}", help=tr("weight_slider_help"),
            )

    b1, b2 = st.columns(2)
    with b1:
        if st.button(tr("save_weights_btn"), width="stretch", type="primary"):
            total = sum(sliders.values()) or 1.0
            save_weights({f: sliders[f] / total for f in FACTORS}, note_key="note_manual")
            st.success(tr("weights_saved"))
            st.rerun()
    with b2:
        if st.button(tr("reset_weights_btn"), width="stretch"):
            save_weights(dict(DEFAULT_WEIGHTS), note_key="note_reset")
            st.success(tr("weights_reset"))
            st.rerun()

    st.markdown(f"##### {tr('current_weights_header')}")
    active = get_latest_weights()
    wdf = pd.DataFrame({
        tr("col_factor"): [factor_name(f) for f in FACTORS],
        tr("col_weight"): [f"{active[f]:.1%}" for f in FACTORS],
        tr("col_change"): [f"{active[f] - DEFAULT_WEIGHTS[f]:+.1%}" for f in FACTORS],
    })
    st.dataframe(wdf, width="stretch", hide_index=True)

    with st.expander(tr("formula_header")):
        st.write(tr("formula_body"))

# ----------------------------------------------------------------------------
# Main Application Controller Setup
# ----------------------------------------------------------------------------
def main() -> None:
    if not st.session_state.get("_db_ready"):
        init_db()
        st.session_state["_db_ready"] = True
    inject_css()

    # Sidebar Interface Controller Layout
    # Language menu first, so changing it re-renders the whole UI in the chosen language.
    lang_choice = st.sidebar.selectbox("🌐 Language / 言語", list(LANGUAGES.keys()), key="lang_select")
    st.session_state["lang"] = LANGUAGES[lang_choice]

    st.sidebar.title(tr("console_title"))
    st.sidebar.caption(tr("console_sub"))

    # MLIT benchmark console: swaps the price-efficiency factor's denominator
    # from built-in estimates to real transaction medians.
    with st.sidebar.expander(tr("mlit_section")):
        st.caption(tr("mlit_note"))
        # Do NOT prefill the secret: type="password" is cosmetic and the value
        # would travel to the client DOM. Leave blank; fall back to the
        # server-side secret when the field is empty.
        typed_key = st.text_input(tr("mlit_key_label"), value="",
                                  type="password", help=tr("mlit_key_help"), key="mlit_key")
        api_key = typed_key or _get_secret("MLIT_API_KEY", "") or ""
        # Refresh ONE prefecture per click. The full roster is now ~40 wards ×
        # 4 quarters ≈ 160 calls — a very long spinner and unnecessary load in
        # one burst. Scoping per prefecture keeps each refresh short and gentle;
        # benchmarks persist, so prefectures can be refreshed on different days.
        refresh_pref = st.selectbox(tr("mlit_scope_label"), PREFECTURES,
                                    format_func=pref_name, key="mlit_scope")
        if st.button(tr("mlit_refresh_btn"), width="stretch"):
            if not api_key:
                st.warning(tr("mlit_need_key"))
            else:
                ok, err = fetch_mlit_benchmarks(api_key, [refresh_pref])
                if ok:
                    # ok is {city_code: (rate, n)} — label by municipality and cap
                    # the list so a multi-ward refresh doesn't flood the sidebar.
                    shown = [f"{municipality_label(cc)} ¥{rt:,.0f}/m²"
                             for cc, (rt, _n) in list(ok.items())[:5]]
                    if len(ok) > 5:
                        shown.append(f"+{len(ok) - 5}…")
                    st.success(tr("mlit_ok", items=", ".join(shown)))
                if err:
                    # err is {display_label: message}; labels are already readable.
                    shown_err = [f"{label} ({msg})" for label, msg in list(err.items())[:5]]
                    if len(err) > 5:
                        shown_err.append(f"+{len(err) - 5}…")
                    st.warning(tr("mlit_err", items=", ".join(shown_err)))
        st.caption(tr("mlit_cities_cached", n=count_municipality_benchmarks()))
        # Code verification: turns a silent wrong-JIS-code failure into a report.
        if st.button(tr("mlit_verify_btn"), width="stretch"):
            if not api_key:
                st.warning(tr("mlit_need_key"))
            else:
                rep = verify_municipality_codes(api_key, [refresh_pref])
                for pf, res in rep.items():
                    if res["error"]:
                        st.warning(f"{pref_name(pf)}: {res['error']}")
                        continue
                    bad = res["unknown"] + res["name_mismatch"]
                    if not bad:
                        st.success(tr("mlit_verify_ok", pref=pref_name(pf), n=res["checked"]))
                    else:
                        details = "; ".join(
                            [f"{nm}={cd} (not in MLIT list)" for nm, cd in res["unknown"][:6]]
                            + [f"{cd}: ours={o} / MLIT={t}" for cd, o, t in res["name_mismatch"][:6]])
                        st.error(tr("mlit_verify_bad", pref=pref_name(pf),
                                    n=len(bad), details=details))
        # Per-prefecture baseline the scorer falls back to when a listing's ward
        # has no city-level row yet (get_sqm_rate(None, pref=...) → tier 2/3).
        for pf in PREFECTURES:
            rate, src, n_samples = get_sqm_rate(None, pref=pf)
            src_label = _benchmark_src_label(src, n_samples)
            st.caption(tr("bench_current", pref=pref_name(pf), rate=rate, src=src_label))

    st.sidebar.caption(tr("persistence_note"))
    if _WAL_FALLBACK:
        st.sidebar.warning(tr("wal_warning"))

    # Four tabs, every one backed by real data. The previous build opened on two
    # tabs of randomly generated listings; analysing a listing you are actually
    # considering is the job, so that is what the app opens on now.
    t1, t2, t3, t4 = st.tabs([tr("tab_analyze"), tr("tab_audit"),
                              tr("tab_market"), tr("tab_weights")])

    with t1: render_manual_analyzer()
    with t2: render_portfolio_audit()
    with t3: render_market_view()
    with t4: render_weights_customizer()

if __name__ == "__main__":
    main()
