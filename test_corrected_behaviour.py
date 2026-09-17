"""Focused regression checks for the corrected integration paths."""
import math
import real_estate_app as app

def main():
    base = app.compute_net_yield(
        36_000_000, float('nan'), 41.64, '1LDK Apartment',
        14_800, 'full', monthly_rent_yen=134_000,
        market_rent_yen=150_000,
    )
    assert base['rent_source'] == 'contract'
    assert base['annual_rent'] == 1_608_000
    assert not base['rent_conflict']

    changed_market = app.compute_net_yield(
        36_000_000, float('nan'), 41.64, '1LDK Apartment',
        14_800, 'full', monthly_rent_yen=134_000,
        market_rent_yen=999_999,
    )
    assert changed_market['noi'] == base['noi']

    fallback = app.compute_net_yield(
        36_000_000, 4.47, 41.64, '1LDK Apartment', 14_800, 'full'
    )
    assert fallback['rent_source'] == 'gross_yield'

    conflict = app.compute_net_yield(
        36_000_000, 6.50, 41.64, '1LDK Apartment',
        14_800, 'full', monthly_rent_yen=134_000,
    )
    assert conflict['rent_conflict']

    for anchor in (15, 18, 22, 25, 30, 35, 40, 50):
        values = [app.size_prior(anchor - 0.1), app.size_prior(anchor),
                  app.size_prior(anchor + 0.1)]
        assert max(values) - min(values) < 1.5, (anchor, values)

    roles = app.map_company_roles([
        ('管理会社', '株式会社エステム管理サービス'),
    ])
    assert roles['management_company'] == '株式会社エステム管理サービス'
    assert not roles.get('agency_name')

    assert 'size_resilience' not in app.FACTORS
    print('CORRECTED BEHAVIOUR: clean')
    print(f"  offer example net yield: {base['net_yield_pct']:.2f}%")
    print(f"  offer example NOI: {base['noi']:.0f}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
