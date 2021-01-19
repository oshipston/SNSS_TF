def SNSSExportDistrictDat(df, SIMDCalc=False):
    i = 0
    SIMDDat = pd.read_table('raw_data/geoData/SIMD16_postcode.txt', index_col=0)
    SIMDDict = dict(zip(SIMDDat.index.values,
                        SIMDDat['SIMD16_Quintile'].values))
    df['T0_SIMD16'] = np.nan
    for p in df['Postcode']:
        if (p == ''):
            df['Postcode'].iloc[i] = np.nan
            df['T0_SIMD16'].iloc[i] = np.nan
            i = i + 1
        else:
            # df['Postcode'].iloc[i] = p.split()[0]
            try:
                df['T0_SIMD16'].iloc[i] = SIMDDict[p]
            except KeyError as err:
                df['T0_SIMD16'].iloc[i] = np.nan
            i = i + 1

    districtT0TotalN = {}
    districtT0FunctionalN = {}
    districtT0HADS = {}
    districtT2PoorCGIN = {}
    districtT2DataN = {}
    districtOutcomeRatio = {}

    for D in np.unique(df['Postcode'][df['Postcode'].notna()]):
        districtT0TotalN[D] = df['ExpGroups'][df['Postcode'] == D].shape[0]
        districtT0FunctionalN[D] = np.sum((df['ExpGroups'][df['Postcode'] == D]) == 1)
        districtT0HADS = np.mean(df['T0_HADS'][df['Postcode'] == D])
        districtT2PoorCGIN[D] = np.sum((df['T2_HealthChange'][df['Postcode'] == D]) < 3)
        districtT2DataN[D] = np.sum((df['T2_HealthChange'][df['Postcode'] == D]).notna())
        districtOutcomeRatio[D] = np.round(districtT2PoorCGIN[D]/districtT2DataN[D], 4)

    SNSSGeoDat = pd.concat([pd.Series(districtT0TotalN),
                            pd.Series(districtT0FunctionalN),
                            pd.Series(districtT0HADS),
                            pd.Series(districtT2PoorCGIN),
                            pd.Series(districtT2DataN),
                            pd.Series(districtOutcomeRatio)], axis=1)
    SNSSGeoDat.columns = ['T0_N', 'T0_functional', 'T0_HADS', 'T2_poorCGI',
                          'T2_DataN', 'T2_poorCGIRatio']
    SNSSGeoDat.to_csv('raw_data/geoData/SNSSDistricts.txt', sep='\t')
    if SIMDCalc:
        print('Saving SNSS SIMD16 Data')
        df['T0_SIMD16'].to_csv('raw_data/geoData/SNSSSIMD16Dat.txt', sep='\t')
    else:
        df['T0_SIMD16'] = pd.read_table('raw_data/geoData/SNSSSIMD16Dat.txt', index_col=0)
    return df

def SNSSFigure_OutcomeStackedBars(df, returnPlot=False):
    """ Assess outcome measure for each group """
    groups = df.groupby('ExpGroups')
    fig = plt.figure(num=1, figsize=(6, 5), dpi=200, frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    pltcol = [list(Color('#950000').range_to(Color('#FF8888'), 5)),
              list(Color('#00336E').range_to(Color('#7DBAFF'), 5))]
    measures = ['T2_SymptomsChange', 'T2_HealthChange']
    healthChange = ['Much Worse', 'Worse', 'Same', 'Better', 'Much Better']
    gLabels = {"1": 'Unexplained', "2": 'Explained'}
    gLabels['1']
    w = 0.8
    x = 0
    y = []
    xLabels = []
    for M in measures:
        g = 0
        for group in groups:
            _, count = np.unique(group[1][M].dropna(), return_counts=True)
            y.append(np.hstack([count/group[1][M].dropna().size, 1, group[1][M].dropna().size]))
            ax.bar(x, y[x][0], w, color=pltcol[g][0].rgb)
            ax.bar(x, y[x][1], w, color=pltcol[g][1].rgb, bottom=np.sum(y[x][0:1]))
            ax.bar(x, y[x][2], w, color=pltcol[g][2].rgb, bottom=np.sum(y[x][0:2]))
            ax.bar(x, y[x][3], w, color=pltcol[g][3].rgb, bottom=np.sum(y[x][0:3]))
            ax.bar(x, y[x][4], w, color=pltcol[g][4].rgb, bottom=np.sum(y[x][0:4]))
            for s in range(0, len(y[x])-2):
                t = ax.text(x, np.mean([np.sum(y[x][0:s]), np.sum(y[x][0:s+1])]), healthChange[s],
                            ha='center',
                            fontsize=6,
                            va='center')
            x = x+1
            g = g+1
            xLabels.append(gLabels.get(str(int(group[0]))) + '\n' + M)
    ax.set_title('Outcomes in the SNSS')
    ax.set_xlabel('Explained By Organic Disease')
    ax.set_ylabel('Proportion')
    ax.set_xticks(np.linspace(0, x-1, x))
    ax.set_xticklabels(xLabels, rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('output/1_PrimaryOutcomeStackedBarsbyGroup.pdf', dpi=200,
                format='pdf', pad_inches=0.1, bbox_inches='tight')
    # mpl_fig
    t = []
    for a in y:
        t.append(a.tolist())
    table = pd.DataFrame(t, columns=healthChange + ['p', 'N'],
                         index=[['T2 IPS']*2 + ['T2 CGI']*2,
                                ['Unexplained', 'Explained']*2]).drop(['p'], axis=1)
    # Table Export
    table.to_csv('output/1_PrimaryOutcomeStackedBarsDat.tsv', sep='\t')
    if returnPlot:
        return fig, table
    else:
        plt.close()
        return table


def SNSSFigure_OutcomeSingleBars(df, returnPlot=False):
    groupVar = 'ExpGroups'
    fig = plt.figure(num=1, figsize=(4, 4), dpi=200, frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    measure = ['T1_poorCGI', 'T2_poorCGI']
    healthChange = ['Much Worse', 'Worse', 'Same', 'Better', 'Much Better']
    w = 0.8
    x = [0, 3]
    y = []
    gCols = list(Color('#b92b27').range_to(Color('#1565C0'), len(np.unique(df[groupVar]))))

    T1ct = pd.crosstab(index=df[groupVar][df[measure[0]].notna()],
                       columns=[df[measure[0]][df[measure[0]].notna()]],
                       margins=True, normalize='index')
    ax.bar(x[0]-0.5, T1ct.iloc[0, 1], w, color=gCols[0].rgb)
    ax.bar(x[0]+0.5, T1ct.iloc[1, 1], w, color=gCols[1].rgb)
    ax.text(x[0]-0.5, T1ct.iloc[0, 1]+0.02, np.round(100*T1ct.iloc[0, 1], decimals=1),
            ha='center',
            fontsize=8,
            va='center')
    ax.text(x[0]+0.5, T1ct.iloc[1, 1]+0.02, np.round(100*T1ct.iloc[1, 1], decimals=1),
            ha='center',
            fontsize=8,
            va='center')
    T2ct = pd.crosstab(index=df[groupVar][df[measure[1]].notna()],
                       columns=[df[measure[1]][df[measure[1]].notna()]],
                       margins=False, normalize='index')
    ax.bar(x[1]-0.5, T2ct.iloc[0, 1], w, color=gCols[0].rgb)
    ax.bar(x[1]+0.5, T2ct.iloc[1, 1], w, color=gCols[1].rgb)
    ax.text(x[1]-0.5, T2ct.iloc[0, 1]+0.02, np.round(100*T2ct.iloc[0, 1], decimals=1),
            ha='center',
            fontsize=8,
            va='center')
    ax.text(x[1]+0.5, T2ct.iloc[1, 1]+0.02, np.round(100*T2ct.iloc[1, 1], decimals=1),
            ha='center',
            fontsize=8,
            va='center')
    ax.set_title('Prevalence of Poor Outcome (CGI)')
    ax.set_xlabel('Explained By Organic Disease')
    ax.set_ylabel('Proportion')
    ax.set_xticks(x)
    ax.set_xticklabels(['3 Months', '12 Months'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([0, 1])
    # fig
    fig.savefig('output/1_PrimaryOutcomesSingleBarsbyGroup.pdf', dpi=200,
                format='pdf', pad_inches=0.1, bbox_inches='tight')

    t = []
    for a in y:
        t.append(a.tolist())
    table = pd.DataFrame(t, columns=healthChange + ['p', 'N'],
                         index=[['T2 IPS']*2 + ['T2 CGI']*2,
                                ['Unexplained', 'Explained']*2]).drop(['p'], axis=1)
    # Table Export
    table.to_csv('output/1_OutcomeBarSingleDat.tsv', sep='\t')
    if returnPlot:
        return fig, table
    else:
        plt.close()
        return table


def SNSSFigure_OutcomeLines(df, returnPlot=False):
    SNSSvars = loadJSON('raw_data/SNSS_vars.json')
    groupVar = 'ExpGroups'
    rowDict = dict(zip(SNSSvars[groupVar]['values'],
                       SNSSvars[groupVar]['valuelabels']))
    primaryOutcomes = [['T1_poorCGI', 'T2_poorCGI'], ['T1_poorIPS', 'T2_poorIPS']]
    gCols = list(Color('#b92b27').range_to(Color('#1565C0'), len(np.unique(df[groupVar]))))

    # Figure 1 Plotting
    fig1 = plt.figure(num=1, figsize=(3, 5), dpi=200, frameon=False)
    ax = fig1.add_subplot(1, 1, 1)
    for O in primaryOutcomes:
        colDict = dict(zip(SNSSvars[O[1]]['values'],
                           SNSSvars[O[1]]['valuelabels']))
        T1ct = pd.crosstab(index=df[groupVar], columns=df[O[0]],
                           margins=False, normalize=False)
        T1ct.columns = [[O[0]]*2, [colDict[int(col)] for col in T1ct.columns]]
        T1ct = T1ct.rename(rowDict)

        T2ct = pd.crosstab(index=df[groupVar], columns=df[O[1]],
                           margins=False, normalize=False)
        T2ct.columns = [[O[1]]*2, [colDict[int(col)] for col in T2ct.columns]]
        T2ct = T2ct.rename(rowDict)    # T2ct
        for G in range(0, len(np.unique(df[groupVar]))):
            T1p = T1ct.iloc[G, 1]/T1ct.iloc[G].sum()
            T1CILow, T1CIUpp = proportion_confint(count=T1ct.iloc[G, 1],
                                                  nobs=T1ct.iloc[G].sum(),
                                                  alpha=0.05)
            T2p = T2ct.iloc[G, 1]/T2ct.iloc[G].sum()
            T2CILow, T2CIUpp = proportion_confint(count=T2ct.iloc[G, 1],
                                                  nobs=T2ct.iloc[G].sum(),
                                                  alpha=0.05)
            x = np.array([0, 3, 12])
            y = np.array([0, T1p, T2p])*-1
            # print([str(T1p) + ' ' + str(T2p)])
            ax.plot(x, y, ls='solid', lw=1.5, marker='None', color=gCols[G].rgb)
            ax.plot([3, 3], np.array([T1CILow, T1CIUpp])*-1, ls='solid',
                    lw=1, marker='_', ms='5', color=gCols[G].rgb)
            ax.plot([12, 12], np.array([T2CILow, T2CIUpp])*-1, ls='solid',
                    lw=1, marker='_', ms='5', color=gCols[G].rgb)
    primaryOutcomesT = pd.concat([T1ct, T2ct], axis=1, sort=False)
    ax.set_title('Prevalence of Poor IPS or CGI Outcome', fontsize=8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Proportion')
    ax.set_xticks([0, 3, 12])
    ax.set_xticklabels(['Baseline', '3 Months', '12 Months'], fontsize=6)
    ax.set_yticks(np.linspace(-1, 0, 6))
    ax.set_yticklabels([1, 0.8, 0.6, 0.4, 0.2, 0], fontsize=6)
    ax.set_ylim([-1, 0])
    ax.set_xlim([0, 13])
    fig1.savefig('output/1_PrimaryOutcomeLinesbyGroup.pdf', dpi=200,
                 format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()
    # Figure 2 Plotting
    groupVar = 'ExpGroups'
    rowDict = dict(zip(SNSSvars[groupVar]['values'],
                       SNSSvars[groupVar]['valuelabels']))
    fig2 = plt.figure(num=1, figsize=(3, 4), dpi=200, frameon=False)
    ax = fig2.add_subplot(1, 1, 1)
    secondaryOutcomes = [['T0_SF12_NormedMCS', 'T1_SF12_NormedMCS', 'T2_SF12_NormedMCS'],
                         ['T0_SF12_NormedPCS', 'T1_SF12_NormedPCS', 'T2_SF12_NormedPCS'],
                         ['T0_PHQNeuro22_Total', 'T1_PHQNeuro22_Total', 'T2_PHQNeuro22_Total'],
                         ['T0_HADS', 'T1_HADS', 'T2_HADS']]
    frames = []
    for O in secondaryOutcomes:
        for G in np.unique(df.ExpGroups).astype(int):
            x = [0, 3, 12]
            T0 = DescrStatsW(df[O[0]].loc[(df.ExpGroups == G) & (df[O[0]].notna())])
            T1 = DescrStatsW(df[O[1]].loc[(df.ExpGroups == G) & (df[O[1]].notna())])
            T2 = DescrStatsW(df[O[2]].loc[(df.ExpGroups == G) & (df[O[2]].notna())])
            y = np.array([T0.mean, T1.mean, T2.mean])
            ax.plot(x, y, ls='solid', lw=1.5, marker='None', color=gCols[G-1].rgb)

            ax.plot([0, 0], T0.tconfint_mean(), ls='solid',
                    lw=1, marker='_', ms='5', color=gCols[G-1].rgb)
            ax.plot([3, 3], T1.tconfint_mean(), ls='solid',
                    lw=1, marker='_', ms='5', color=gCols[G-1].rgb)
            ax.plot([12, 12], T2.tconfint_mean(), ls='solid',
                    lw=1, marker='_', ms='5', color=gCols[G-1].rgb)
            LCI = np.array([T0.tconfint_mean()[0], T1.tconfint_mean()[0], T2.tconfint_mean()[0]])
            UCI = np.array([T0.tconfint_mean()[1], T1.tconfint_mean()[1], T2.tconfint_mean()[1]])
            type(y)
            type(LCI)
            frames.append(pd.DataFrame(np.reshape(np.append(y, [LCI, UCI]), newshape=[1, 9]),
                                       index=[[O[0][3:]], [rowDict[G]]]))
            [O[0][3:], rowDict[G]]
    secondaryOutcomesT = pd.concat(frames)
    secondaryOutcomesT.columns = [['mean']*3 + ['LCI']*3 + ['UCI']*3, ['T0', 'T1', 'T2']*3]
    ax.set_title('Mean Secondary Measures')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Scores')
    ax.set_xticks([0, 3, 12])
    ax.set_xticklabels(['Baseline', '3 Months', '12 Months'], fontsize=6)
    # ax.set_ylim([0, 50])
    ax.set_xlim([-0.5, 13])
    fig2.savefig('output/1_SecondaryOutcomeLinesbyGroup.pdf', dpi=200,
                 format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()
    # Figure Export
    fig = {'primaryOutcomes': fig1,
           'secondaryOutcomes': fig2}
    table = {'primaryOutcomes': primaryOutcomesT,
             'secondaryOutcomes': secondaryOutcomesT}

    if returnPlot:
        return fig, table
    else:
        return table
