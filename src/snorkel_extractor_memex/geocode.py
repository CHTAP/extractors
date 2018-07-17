import googlemaps as gm
import gmaps
import pandas as pd
import numpy as np
from shapely.geometry import MultiPoint
import warnings
warnings.filterwarnings('ignore')
maps_api_key = 'AIzaSyA0Veo5Lc6JOwDjNgQvPEhQB4AiZcrYQGI'
gmaps.configure(api_key=maps_api_key)

def get_possible_locations(plc):
    """
    INPUTS
    plc: string describing place to match

    OUTPUTS
    jsn: full json structure returned from API call
    plcs: list of candidate location strings
    """ 
    api_key = 'AIzaSyDbk3lLZHuQVKDRBN99_oz-p4AJjIzhA0w'
    gms = gm.Client(key=api_key)
    qo = gm.places.places_autocomplete(gms,plc)
    cl = [a['description'] for a in qo]
    return qo,cl

def get_geocode(plc):
    """
    INPUTS
    plc: string describing place to match

    OUTPUTS
    jsn: full json structure returned from API call
    plcs: list of candidate location strings
    """
    api_key = 'AIzaSyBlLyOaasYMgMxFGUh2jJyxIG0_pZFF_jM'
    gms = gm.Client(key=api_key)
    qo = gm.geocoding.geocode(gms,plc)
    lat = qo[0]['geometry']['location']['lat']
    lng = qo[0]['geometry']['location']['lng']
    return qo,(lat,lng)

def slice_pd_by_cont(dfm,col,val,pres=True,lower=False,union=False):
    """
    Returns dataframe where column values include/exclude values in provided list
    
    INPUTS:
    dfm: dataframe
    col: column header
    val: list of strings to include/ignore
    pres: true to include, false to exclude
    union: include union of these values
    """
    if union:
        val = ['|'.join(val)]
    for vl in val:
        if ~lower:
            if pres:
                dfm = dfm.loc[dfm[col].str.contains(vl,na=False)]
            else:
                dfm = dfm.loc[~dfm[col].str.contains(vl,na=False)]
        else:
            if pres:
                dfm = dfm.loc[dfm[col].str.lower().str.contains(vl,na=False)]
            else:
                dfm = dfm.loc[~dfm[col].str.lower().str.contains(vl,na=False)]
    return dfm

def map_candidates_and_centroid(dfm):
    """
    INPUT
    dfm: dataframe containing at least latitude, longitude
    
    OUTPUT
    centroid: np array of lat/lon of location centroid
    """
    df_cans = dfm
    df_cans_map = dfm[['latitude','longitude']]
    df_cans['lat_long'] = df_cans[['latitude', 'longitude']].apply(tuple, axis=1)
    point_tup_lst = df_cans['lat_long'].tolist()
    points = MultiPoint(point_tup_lst)
    cent = np.array(points.centroid)
    cent_df = pd.DataFrame([cent]) #this is a rough centroid estimate
    fig = gmaps.Map()
    can_layer = gmaps.symbol_layer(
    df_cans_map, fill_color="green", stroke_color="green", scale=2)
    cent_layer = gmaps.symbol_layer(
    cent_df, fill_color="red", stroke_color="red", scale=2)
    fig.add_layer(can_layer)
    fig.add_layer(cent_layer)
    fig
    return cent,fig
