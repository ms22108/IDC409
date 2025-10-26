def feature_dist(feature_to_plot, title_add=None, both=None, x_axis_lim = None, y_log = None):
    # function for visualizing features
    '''
    ____________________
    Args
        feature_to_plot:
        title_add (string, optional): additional description to the plot title
        both (boolean, optional): True: histogram and kde, False: histogram only
        x_axis_lim (list, optional): set the x axis limits manually
        y_log (boolean, optional): logarithmic y axis scale
    ____________________
    
    '''

    bin_size = 50
    plt.figure(figsize=(5,3))   
    sns.histplot(df[df['is_signal'] == 0][feature_to_plot], 
                 label='Background', color='red', alpha=0.5,
                 stat='density', bins=bin_size, element='step')
    sns.histplot(df[df['is_signal'] == 1][feature_to_plot], 
                 label='Signal', color='blue', alpha=0.5, 
                 stat='density', bins=bin_size, element='step')

    if (title_add is None):
        title_add = ''
    
    plt.title(feature_to_plot + ' ' + str(title_add))
    plt.legend(loc='upper left')
    
    # setting x axis limits
    if x_axis_lim == None: plt.xlim(min(df[feature_to_plot]), max(df[feature_to_plot])) 
    else: plt.xlim(x_axis_lim)

    # setting y axis scale
    if y_log == True: plt.yscale('log')   
    # plotting a continuous distribution
    if (both is True):
        plt.figure(figsize=(5,5))    
        sns.displot(df, x=feature_to_plot, hue="is_signal", hue_order = [1, 0], kind="kde", fill=True)
        plt.title(feature_to_plot + ' ' + str(title_add))

    # setting x axis limits
    if x_axis_lim == None: plt.xlim(min(df[feature_to_plot]), max(df[feature_to_plot])) 
    else: plt.xlim(x_axis_lim)

    # setting y axis scale
    if y_log == True:
        plt.yscale('log')