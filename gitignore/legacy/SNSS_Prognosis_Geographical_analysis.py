import os
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches
import os
os.getcwd()
SNSSDat = pd.read_table('raw_data/SNSSDistricts.txt', index_col=0)
popDat = pd.read_table('raw_data/2011_postalDistrictPopulations.txt', index_col=0)
districtMap = gpd.read_file('raw_data/geoData/spd-district-boundaries-18-1/District_18_1.shp')
dmapPoorOutcome = []
for D in districtMap['District']:
    if (D in SNSSDat.index):
        fxd = SNSSDat.loc[D]['ratioPoorOutcome']
        if ~np.isnan(fxd):
            dmapPoorOutcome.append(fxd)
        else:
            dmapPoorOutcome.append(np.nan)  # If no follow up add NaN
    else:
        dmapPoorOutcome.append(np.nan)  # If no patients at all from area add NaN
districtMap['poorOutcome'] = dmapPoorOutcome
# SNSSDat = SNSSDat.drop(['AB36'])
dmapFxPrevalence = []
for D in districtMap['District']:
    if (D in SNSSDat.index) & (D in popDat.index):
        fxN = SNSSDat.loc[D]['NFunctional']
        pop = popDat.loc[D]['N']
        dmapFxPrevalence.append((fxN/pop)*1000)
    else:
        dmapFxPrevalence.append(np.nan)  # If no patients at all from area add NaN
districtMap['FxPrevalence'] = dmapFxPrevalence
districtMap = districtMap.to_crs(epsg=4326)

districtEdges = gpd.read_file('raw_data/geoData/spd-district-boundaries-18-1/District_18_1.shp')
districtEdges = districtMap.to_crs(epsg=4326)

landMap = gpd.read_file('raw_data/geoData/ne_10m_admin_0_map_units/ne_10m_admin_0_map_units.shp')
# landMap.crs
water = patches.Rectangle([-8, 50], 20, 20, fc='xkcd:blue grey', ec=None, zorder=-1)

fig = plt.figure(num=1, figsize=(5, 5), dpi=100, frameon=False)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.add_patch(water)
englandBound = landMap[landMap.NAME_EN == 'England']
_ = englandBound.plot(ax=ax, facecolor='xkcd:charcoal', lw=0.5, edgecolor='xkcd:charcoal')
_ = districtMap.plot(ax=ax, column='FxPrevalence',
                     edgecolor='none', cmap='Blues', legend=True)

_ = districtEdges.plot(ax=ax, facecolor='none', lw=0.1,
                       edgecolor='xkcd:grey', alpha=0.2)

ax.set_extent([-8, 0, 54.5, 61.3], crs=ccrs.Robinson())  # Scotland
# ax.set_extent([-3.3, -3, 55.8, 56], crs=ccrs.Robinson())  # Edinburgh
ax.set_aspect(1.5)
leg = ax.get_legend()
# leg.set_bbox_to_anchor((0., 0., 0.2, 0.2))
fig

# geoData/scotland_hba_2001/scotland_hba_2001
# geoData/spd-district-boundaries-18-1/District_18_1
# geoData/Scotland_laulevel2_2011_clipped/scotland_laulevel2_2011_clipped
# geoData/Scotland_groslcls_2001/scotland_groslcls_2001

fig.savefig('output/FunctionalDisorderPrevalenceMap.pdf', dpi=100,
            format='pdf', pad_inches=0.1, bbox_inches='tight')


# df = pd.read_table('raw_data/2011_postalSectorPopulations.txt', header=None,
#                    names=['Sector', 'N'])
# df['District'] = [str.split()[0] for str in df['Sector']]
# districtPop = {}
# for d in np.unique(df['District']):
#     idx = df['District'].str.find(d)
#     districtPop[d] = np.sum(df['N'][(idx == 0)])
# t = pd.DataFrame.from_dict(districtPop, orient='index', columns=['N'])
# t.to_csv('raw_data/2011_postalDistrictPopulations.txt', sep='\t')
