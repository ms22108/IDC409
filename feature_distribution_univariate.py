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