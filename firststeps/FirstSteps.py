import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings

class FirstSteps(object):
    
    def jupyter_settings():#FIRST STEPS
        plt.style.use( 'bmh' )
        plt.rcParams['figure.figsize'] = [24, 12]
        plt.rcParams['font.size'] = 24
        
        pd.options.display.max_columns = 50
        pd.options.display.max_rows = 50
        pd.set_option( 'display.expand_frame_repr', False )
        
        sns.set()
        sns.set_theme(palette = 'rocket')
        
        warnings.filterwarnings('ignore')
    
#####################################################################################################################################################

    def num_statistic(data): # para estatisticas descritivas #FIRST STEPS
            
        #seleção de variáveis numéricas
        num_attributes = data.select_dtypes(include = ('int64', 'float64'))
    
        #tendencia Central - mean, median ----- dispersão - std, min, max, range, skew, kurtosis
        count_ = pd.DataFrame( num_attributes.count()).T
        range_ = pd.DataFrame( num_attributes.apply( lambda x: x.max() - x.min() ) ).T 
        min_ = pd.DataFrame( num_attributes.min()).T
        q1 = pd.DataFrame( num_attributes.quantile(0.25)).T
        median_ = pd.DataFrame( num_attributes.apply(np.median)).T
        q3 = pd.DataFrame( num_attributes.quantile(0.75)).T
        max_ = pd.DataFrame( num_attributes.max()).T
        mean_ = pd.DataFrame( num_attributes.apply(np.mean)).T
        std_ = pd.DataFrame( num_attributes.apply(np.std)).T
        skew = pd.DataFrame( num_attributes.apply( lambda x: x.skew() ) ).T 
        kurtosis = pd.DataFrame( num_attributes.apply( lambda x: x.kurtosis() ) ).T 
    
        #Concatenar
        m = pd.concat( [count_, range_, min_, q1, median_, q3, max_, mean_, std_, skew, kurtosis] ).T.reset_index()
        m.columns = ['attributes', 'count', 'range', 'min', '25%', '50%', '75%', 'max', 'mean', 'std', 'skew', 'kurtosis']
        
        return m
    
    #####################################################################################################################################################
        
    def type_na(data): #FIRST STEPS
        type_na= pd.DataFrame({'features': data.columns, 
                               'type': data.dtypes,
                               'soma_nulos': data.isna().sum(), 
                               'perc_nulos': data.isna().mean()}).reset_index(drop=True)
        
        return type_na
    
    #####################################################################################################################################################   
    
    def cat_features_plots(data): #FIRST STEPS
        cat = data.select_dtypes(include='object')
        for i in cat.columns:
            fig,ax = plt.subplots()
            sns.countplot(cat, x=i, ax=ax)
            ax.set_title(f'Contagem de {i}')
            
            for p in ax.patches:
                h = p.get_height()
                ax.text(p.get_x() + (p.get_width()/2), h , '{:.1f}%'.format(h/len(cat)*100), ha='center')
    
        return None
    
    #####################################################################################################################################################
    
    def num_features_plot(data): #FIRST STEPS
        num_attributes = data.select_dtypes(include = ('int64', 'float64'))
        num_attributes.hist(bins=25, color='crimson');
        
    #####################################################################################################################################################
    
    def convertion_plots(df1, feature): #FIRST STEPS
        sns.set_style("darkgrid", {'axes.grid' : False})
        
        total = df1.groupby(feature)[['id']].count().reset_index()
        response1 = (df1.query('response == 1')
                        .groupby(feature)[['id']]
                        .count()
                        .reset_index()
                        .rename(columns={'id': 'interessed'}))
        total = total.merge(response1, how='left', on=feature).fillna(0)
        total['perc'] = ((total.interessed/total.id)*100).astype(int)
        
        max_total = total.id.max() * 1.1
        
        fig, ax = plt.subplots()
        sns.barplot(total, x=feature, y='id', ax=ax, palette='Set2_r')
        plt.ylim(0,max_total)
        ax1=ax.twinx()
        for i in total.index:
            label = ' ' + str(total.loc[i, 'perc']) + '%'
            y = (total.loc[i, 'interessed'])
            plt.annotate(label, (i,y), ha = 'center', va='bottom', rotation= 55)
            
        sns.barplot(total, x=feature, y='interessed', ax=ax1, palette=['black'], alpha=0.5);
        plt.ylim(0,max_total)
        
        return None
    
    #####################################################################################################################################################
    
    