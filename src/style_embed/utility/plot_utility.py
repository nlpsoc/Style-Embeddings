def init_plot():
    import matplotlib
    from matplotlib import pyplot as plt
    plt.ioff()
    plt.rcParams.update({'font.size': 26})
    matplotlib.use('Agg')
