import pandas as pd


def create_detailed_date_features(df, date_col='date'):
    df = df.copy()
    date_series = pd.to_datetime(df[date_col])

    df['year'] = date_series.dt.year
    df['month'] = date_series.dt.month
    df['day'] = date_series.dt.day
    df['quarter'] = date_series.dt.quarter
    df['dayofweek'] = date_series.dt.dayofweek
    df['weekofyear'] = date_series.dt.isocalendar().week.astype(int)
    df['dayofyear'] = date_series.dt.dayofyear
    df['isweekend'] = df['dayofyear'].isin([5, 6]).astype(int)

    df['ismonthstart'] = date_series.dt.is_month_start.astype(int)
    df['ismonthend'] = date_series.dt.is_month_end.astype(int)
    df['isquarterstart'] = date_series.dt.is_quarter_start.astype(int)
    df['isquarterend'] = date_series.dt.is_quarter_end.astype(int)
    df['isyearstart'] = date_series.dt.is_year_start.astype(int)
    df['isyearend'] = date_series.dt.is_year_end.astype(int)
    df['weekofmonth'] = (date_series.dt.day - 1) // 7 + 1

    month_to_season = {
        12: 'Winter',
        1: 'Winter',
        2: 'Winter',
        3: 'Spring',
        4: 'Spring',
        5: 'Spring',
        6: 'Summer',
        7: 'Summer',
        8: 'Summer',
        9: 'Autumn',
        10: 'Autumn',
        11: 'Autumn'}
    df['season'] = df['month'].map(month_to_season)

    return df


def create_missingness_flags(df, cols_to_flag_missing):
    df_processed = df.copy()
    for col in cols_to_flag_missing:
        if col in df_processed.columns:
            df_processed[f'hasmissing_{col}'] = df_processed[col].isnull().astype(int)
            df_processed[col] = df_processed[col].fillna('Unknown')
    return df_processed


def categorize_transactions(df: pd.DataFrame,
                            column_name: str) -> pd.DataFrame:

    category_map = {
        'Sell': 'Standard Sale',
        'Sell - Pre registration': 'Standard Sale',
        'Delayed Sell': 'Standard Sale',
        'Sale On Payment Plan': 'Standard Sale',

        'Mortgage Registration': 'Mortgage',
        'Modify Mortgage': 'Mortgage',
        'Delayed Mortgage': 'Mortgage',
        'Mortgage Transfer': 'Mortgage',
        'Mortgage Pre-Registration': 'Mortgage',
        'Development Mortgage': 'Mortgage',
        'Mortgage Transfer Pre-Registration': 'Mortgage',
        'Modify Mortgage Pre-Registration': 'Mortgage',
        'Modify Delayed Mortgage': 'Mortgage',

        'Lease to Own Registration': 'Lease Agreement',
        'Lease Finance Registration': 'Lease Agreement',
        'Lease to Own Transfer': 'Lease Agreement',
        'Lease to Own Modify': 'Lease Agreement',
        'Lease to Own Registration Pre-Registration': 'Lease Agreement',
        'Delayed Lease to Own Registration': 'Lease Agreement',
        'Lease Finance Modification': 'Lease Agreement',
        'Lease Development Registration': 'Lease Agreement',
        'Lease to Own on Development Registration': 'Lease Agreement',
        'Lease Development Modify': 'Lease Agreement',
        'Delayed Lease to Own Modify': 'Lease Agreement',
        'Delayed Lease to Own Transfer': 'Lease Agreement',

        'Development Registration': 'Development',
        'Sell Development': 'Development',
        'Delayed Development': 'Development',
        'Grant Development': 'Development',
        'Development Registration Pre-Registration': 'Development',
        'Development Mortgage Pre-Registration': 'Development',
        'Lease to Own on Development Modification': 'Development',
        'Transfer Development Mortgage': 'Development',
        'Portfolio Development Registration': 'Development',
        'Delayed Sell Development': 'Development',
        'Sell Development - Pre Registration': 'Development',
        'Modify Development Mortgage': 'Development',

        'Grant': 'Grant',
        'Grant Pre-Registration': 'Grant',
        'Grant on Delayed Sell': 'Grant',

        'Portfolio Mortgage Registration Pre-Registration': 'Portfolio',
        'Portfolio Mortgage Development Registration': 'Portfolio',
        'Portfolio Mortgage Modification Pre-Registration': 'Portfolio',
        'Portfolio Mortgage Development Modification': 'Portfolio',
        'Delayed Portfolio Mortgage': 'Portfolio',
        'Portfolio Mortgage Registration': 'Portfolio',
        'Portfolio Mortgage Modification': 'Portfolio',
        'Portfolio Mortgage Transfer': 'Portfolio'
    }

    df = df.copy()
    df[column_name +
       '_grouped'] = df[column_name].map(category_map).fillna('Other_Transaction')
    return df


district_mapping = {
    'Hor Al Anz': 'Deira',
    'Naif': 'Deira',
    'Al Ras': 'Deira',
    'Al Muteena': 'Deira',
    'Al Dhagaya': 'Deira',
    'Al Waheda': 'Deira',
    'Abu Hail': 'Deira',
    'Al Murqabat': 'Deira',
    'Al Mararr': 'Deira',
    'Al Rega': 'Deira',
    'Al Baraha': 'Deira',
    'Rega Al Buteen': 'Deira',
    'Al Buteen': 'Deira',
    'Al Sabkha': 'Deira',
    'Al Khabeesi': 'Deira',
    'Eyal Nasser': 'Deira',
    'Hor Al Anz East': 'Deira',
    'Al Mamzer': 'Deira',
    'Port Saeed': 'Deira',
    'Al Kifaf': 'Bur Dubai',
    'Al Suq Al Kabeer': 'Bur Dubai',
    'Mankhool': 'Bur Dubai',
    'Al Raffa': 'Bur Dubai',
    'Al Hamriya': 'Bur Dubai',
    'Al Karama': 'Bur Dubai',
    'Al Jafliya': 'Bur Dubai',
    'Um Hurair Second': 'Bur Dubai',
    'Um Hurair First': 'Bur Dubai',
    'Oud Metha': 'Bur Dubai',
    'Al Hudaiba': 'Bur Dubai',
    'Um Suqaim Third': 'Coastal Strip (Jumeirah/Umm Suqeim)',
    'Um Suqaim Second': 'Coastal Strip (Jumeirah/Umm Suqeim)',
    'Jumeirah First': 'Coastal Strip (Jumeirah/Umm Suqeim)',
    'Al Saffa Second': 'Coastal Strip (Jumeirah/Umm Suqeim)',
    'Al Satwa': 'Coastal Strip (Jumeirah/Umm Suqeim)',
    'Um Suqaim First': 'Coastal Strip (Jumeirah/Umm Suqeim)',
    'Jumeirah Third': 'Coastal Strip (Jumeirah/Umm Suqeim)',
    'Al Saffa First': 'Coastal Strip (Jumeirah/Umm Suqeim)',
    'Um Al Sheif': 'Coastal Strip (Jumeirah/Umm Suqeim)',
    'Jumeirah Second': 'Coastal Strip (Jumeirah/Umm Suqeim)',
    'Al Bada': 'Coastal Strip (Jumeirah/Umm Suqeim)',
    'Al Manara': 'Coastal Strip (Jumeirah/Umm Suqeim)',
    'Al Wasl': 'Coastal Strip (Jumeirah/Umm Suqeim)',
    'Business Bay': 'Downtown Dubai & Business Bay',
    'Burj Khalifa': 'Downtown Dubai & Business Bay',
    'Zaabeel Second': 'Downtown Dubai & Business Bay',
    'Zaabeel First': 'Downtown Dubai & Business Bay',
    'Trade Center Second': 'Downtown Dubai & Business Bay',
    'Trade Center First': 'Downtown Dubai & Business Bay',
    'Hadaeq Sheikh Mohammed Bin Rashid': 'Meydan & Nad Al Shiba',
    'Al Merkadh': 'Meydan & Nad Al Shiba',
    'Nad Al Shiba First': 'Meydan & Nad Al Shiba',
    'Nad Al Shiba Third': 'Meydan & Nad Al Shiba',
    'Nad Al Shiba': 'Meydan & Nad Al Shiba',
    'Nad Al Shiba Fourth': 'Meydan & Nad Al Shiba',
    'Nad Al Shiba Second': 'Meydan & Nad Al Shiba',
    'Madinat Latifa': 'Meydan & Nad Al Shiba',
    'Al Barsha South Fourth': 'Al Barsha & Al Quoz',
    'Al Barsha South Fifth': 'Al Barsha & Al Quoz',
    'Al Goze First': 'Al Barsha & Al Quoz',
    'Al Barshaa South Third': 'Al Barsha & Al Quoz',
    'Al Barsha First': 'Al Barsha & Al Quoz',
    'Al Barsha Second': 'Al Barsha & Al Quoz',
    'Al Barsha Third': 'Al Barsha & Al Quoz',
    'Al Goze Fourth': 'Al Barsha & Al Quoz',
    'Al Barshaa South Second': 'Al Barsha & Al Quoz',
    'Al Barshaa South First': 'Al Barsha & Al Quoz',
    'Al Goze Third': 'Al Barsha & Al Quoz',
    'Al Thanyah Third': 'TECOM, Greens & Emirates Hills Area',
    'Al Thanyah Fifth': 'TECOM, Greens & Emirates Hills Area',
    'Al Thanayah Fourth': 'TECOM, Greens & Emirates Hills Area',
    'Al Safouh Second': 'TECOM, Greens & Emirates Hills Area',
    'Al Thanyah First': 'TECOM, Greens & Emirates Hills Area',
    'Al Safouh First': 'TECOM, Greens & Emirates Hills Area',
    'Marsa Dubai': 'Dubai Marina & JBR',
    'Al Khairan First': 'Dubai Marina & JBR',
    'Palm Jumeirah': 'Palm Jumeirah',
    'Al Yelayiss 1': 'Jebel Ali & Dubai South West',
    'Al Kheeran': 'Jebel Ali & Dubai South West',
    'Al Yelayiss 2': 'Jebel Ali & Dubai South West',
    'Jabal Ali First': 'Jebel Ali & Dubai South West',
    'Jabal Ali': 'Jebel Ali & Dubai South West',
    'Jabal Ali Industrial Second': 'Jebel Ali & Dubai South West',
    'Dubai Investment Park Second': 'Jebel Ali & Dubai South West',
    'Dubai Investment Park First': 'Jebel Ali & Dubai South West',
    'Jabal Ali Industrial First': 'Jebel Ali & Dubai South West',
    'Ghadeer Al tair': 'Jebel Ali & Dubai South West',
    'Hessyan First': 'Jebel Ali & Dubai South West',
    'Saih Shuaib 1': 'Jebel Ali & Dubai South West',
    'Saih Shuaib 2': 'Jebel Ali & Dubai South West',
    'Saih Shuaib 4': 'Jebel Ali & Dubai South West',
    'Mena Jabal Ali': 'Jebel Ali & Dubai South West',
    'Al Yelayiss 5': 'Jebel Ali & Dubai South West',
    'Hessyan Second': 'Jebel Ali & Dubai South West',
    'Saih Shuaib 3': 'Jebel Ali & Dubai South West',
    'Wadi Al Safa 5': 'Dubai South / New Developments',
    'Me\'Aisem First': 'Dubai South / New Developments',
    'Wadi Al Safa 6': 'Dubai South / New Developments',
    'Al Hebiah First': 'Dubai South / New Developments',
    'Al Hebiah Third': 'Dubai South / New Developments',
    'Al Hebiah Fourth': 'Dubai South / New Developments',
    'Wadi Al Safa 3': 'Dubai South / New Developments',
    'Wadi Al Safa 7': 'Dubai South / New Developments',
    'Al Hebiah Sixth': 'Dubai South / New Developments',
    'Al Yufrah 1': 'Dubai South / New Developments',
    'Al Hebiah Fifth': 'Dubai South / New Developments',
    'Wadi Al Safa 2': 'Dubai South / New Developments',
    'Madinat Hind 4': 'Dubai South / New Developments',
    'Al Hebiah Second': 'Dubai South / New Developments',
    'Madinat Hind 3': 'Dubai South / New Developments',
    'AL Athbah': 'Dubai South / New Developments',
    'Nazwah': 'Dubai South / New Developments',
    'Al Yufrah 2': 'Dubai South / New Developments',
    'Al Lusaily': 'Dubai South / New Developments',
    'Wadi Al Safa 4': 'Dubai South / New Developments',
    'Saih Aldahal': 'Dubai South / New Developments',
    'Al Rowaiyah First': 'Dubai South / New Developments',
    'Al Eyas': 'Dubai South / New Developments',
    'Me\'Aisem Second': 'Dubai South / New Developments',
    'Al Warsan First': 'Eastern Dubai',
    'Nadd Hessa': 'Eastern Dubai',
    'Al Mizhar Second': 'Eastern Dubai',
    'Al Mizhar Third': 'Eastern Dubai',
    'Al Aweer First': 'Eastern Dubai',
    'Warsan Fourth': 'Eastern Dubai',
    'Mirdif': 'Eastern Dubai',
    'Al Khawaneej Second': 'Eastern Dubai',
    'Al Warqa Third': 'Eastern Dubai',
    'Al Mizhar First': 'Eastern Dubai',
    'Al Warsan Second': 'Eastern Dubai',
    'Nad Al Hamar': 'Eastern Dubai',
    'Al Warqa Fourth': 'Eastern Dubai',
    'Al Warqa First': 'Eastern Dubai',
    'Al Aweer Second': 'Eastern Dubai',
    'Al Khawaneej First': 'Eastern Dubai',
    'Al Warqa Second': 'Eastern Dubai',
    'Wadi Al Amardi': 'Eastern Dubai',
    'Mushrif': 'Eastern Dubai',
    'Al Ttay': 'Eastern Dubai',
    'Muhaisanah Third': 'North-Eastern Dubai',
    'Muhaisanah First': 'North-Eastern Dubai',
    'Al Twar First': 'North-Eastern Dubai',
    'Al Nahda First': 'North-Eastern Dubai',
    'Al Twar Fourth': 'North-Eastern Dubai',
    'Al Nahda Second': 'North-Eastern Dubai',
    'Al Twar Third': 'North-Eastern Dubai',
    'Oud Al Muteena First': 'North-Eastern Dubai',
    'Al Twar Second': 'North-Eastern Dubai',
    'Al Qusais Second': 'North-Eastern Dubai',
    'Al Qusais Industrial Fourth': 'North-Eastern Dubai',
    'Al Qusais First': 'North-Eastern Dubai',
    'Al Qusais Industrial Fifth': 'North-Eastern Dubai',
    'Muhaisanah Fourth': 'North-Eastern Dubai',
    'Muhaisanah Second': 'North-Eastern Dubai',
    'Al Qusais Industrial Third': 'North-Eastern Dubai',
    'Al Qusais Industrial First': 'North-Eastern Dubai',
    'Al Qusais Industrial Second': 'North-Eastern Dubai',
    'Madinat Al Mataar': 'Airport & Nearby Areas',
    'Al Rashidiya': 'Airport & Nearby Areas',
    'Al Garhoud': 'Airport & Nearby Areas',
    'Nad Shamma': 'Airport & Nearby Areas',
    'Um Ramool': 'Airport & Nearby Areas',
    'Al Jadaf': 'Industrial Areas (Central/East)',
    'Ras Al Khor Industrial First': 'Industrial Areas (Central/East)',
    'Ras Al Khor Industrial Third': 'Industrial Areas (Central/East)',
    'Al Goze Industrial First': 'Industrial Areas (Central/East)',
    'Al Goze Industrial Second': 'Industrial Areas (Central/East)',
    'Al Goze Industrial Fourth': 'Industrial Areas (Central/East)',
    'Ras Al Khor Industrial Second': 'Industrial Areas (Central/East)',
    'Bukadra': 'Industrial Areas (Central/East)',
    'Island 2': 'Islands & Special Zones',
    'Madinat Dubai Almelaheyah': 'Islands & Special Zones',
    'World Islands': 'Islands & Special Zones',
    'Palm Deira': 'Islands & Special Zones',
    'Hatta': 'Hatta'}


def add_district_column(
        df,
        area_col_name='area_name_en',
        new_col_name='district'):
    df_processed = df.copy()
    df_processed[new_col_name] = df_processed[area_col_name].map(
        district_mapping)

    unmapped_count = df_processed[new_col_name].isnull().sum()
    if unmapped_count > 0:
        df_processed[new_col_name].fillna('Unknown_District', inplace=True)

    return df_processed
