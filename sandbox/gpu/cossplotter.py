from cosstester import *

def _getDefaultPlotConfig(figure):
    config = {'style': 'ro',
              'plotType': 'plot',
              'min_x': None,
              'max_x': None,
              'min_y': None,
              'max_y': None,
              'xticks': None,
              'yticks': None}
    if figure['x']['type'] == 'double':
        config['style'] = None
        config['plotType'] = 'bar'
        config['min_y'] = 0
        config['xticks'] = 'names'
        return config
    else:
        return config

def plotResults(_file, plotTypes=None, get_stored_fstates=False):
    """
    Example:
    plotTypes=[{'x': {'type': 'block_size'},
                'y': {'type': 'runtime'}},
               {'x': {'type': 'field_parameter_values',
                      'index': 0,
                      'name': 'g_to'},
                'y': {'type': 'field_state_values',
                      'index': 0,
                      'name': 'V'}}]
    """
    title, data = getDataFromFileSimple(_file)
    names = [datum['name'] for datum in data]
    subnames = [name.split(' ', 2)[-1][1:-1] for name in names]
    if plotTypes is None:
        subsubnames = [s.split(',') for s in subnames]
        namekeys, namevalues = zip(*[
            zip(*[s.split('=') for s in ssname])
            for ssname in subsubnames])

        if len(set(namekeys)) != 1:
            print 'Cannot determine what to plot.'
            if len(set(namekeys)) < 16:
                print 'Keys', set(namekeys)
            else:
                print len(set(namekeys)), 'keys'
            return
        else:
            plotTypes = list()
            for _type in list(set(namekeys))[0]:
                if _type == 'um_nodes':
                    _type = 'num_nodes'
                if _type == 'lock_size':
                    _type = 'block_size'
                plotTypes.append({'x': {'type': _type},
                                  'y': {'type': 'runtime'}})

    validPlotTypes = ('block_size', 'double', 'dt', 'field_parameter_values',
                      'field_state_values', 'num_nodes', 'runtime')
    figures = list()
    for pType in plotTypes:
        if pType['x']['type'] not in validPlotTypes \
                or pType['y']['type'] not in validPlotTypes:
            print str(pType) + ' plot not yet implemented'
        else:
            figures.append(pType)

    for figure in figures:
        fig, ax = plt.subplots()

        plotConfig = _getDefaultPlotConfig(figure)
        plotConfig.update(figure)

        xy_pos = dict()

        for axis in ('x', 'y'):
            xy_pos[axis] = None

            if figure[axis]['type'] in \
                    ('block_size', 'dt', 'num_nodes', 'runtime'):
                xy_pos[axis] = [datum[figure[axis]['type']] for datum in data]

            if figure[axis]['type'] == 'field_state_values':
                xy_pos[axis] = [datum['field_state_values'] for datum in data]

            if figure[axis]['type'] == 'field_parameter_values':
                xy_pos[axis] = [datum['field_parameter_values'] for datum in data]

            if figure[axis]['type'] == 'double':
                xy_pos[axis] = [i for i in xrange(len(data))]

        x_pos = xy_pos['x']
        y_pos = xy_pos['y']

        if plotConfig['plotType'] == 'plot':
            if plotConfig['style'] is not None:
                ax.plot(x_pos, y_pos, plotConfig['style'])
            else:
                ax.plot(x_pos, y_pos)
        elif plotConfig['plotType'] == 'bar':
            ax.bar(x_pos, y_pos)

        if plotConfig['xticks'] is None:
            min_x, max_x = min(x_pos), max(x_pos)
            if plotConfig['min_x'] is not None:
                min_x = plotConfig['min_x']
            if plotConfig['max_x'] is not None:
                max_x = plotConfig['max_x']
            ax.set_xlim((min_x, max_x))
        elif plotConfig['xticks'] == 'names':
            plt.xticks(x_pos, names)
        elif plotConfig['xticks'] == 'subnames':
            plt.xticks(x_pos, subnames)
        else:
            plt.xticks(x_pos, plotConfig['xticks'])

        if plotConfig['yticks'] is None:
            min_y, max_y = min(y_pos), max(y_pos)
            if plotConfig['min_y'] is not None:
                min_y = plotConfig['min_y']
            if plotConfig['max_y'] is not None:
                max_y = plotConfig['max_y']
            ax.set_ylim((min_y, max_y))
        elif plotConfig['yticks'] == 'names':
            plt.yticks(y_pos, names)
        elif plotConfig['yticks'] == 'subnames':
            plt.yticks(y_pos, subnames)
        else:
            plt.yticks(y_pos, plotConfig['yticks'])

        plt.title(PLOT_TITLES[figure['y']['type']].format(title))
        plt.xlabel(PLOT_STRINGS[figure['x']['type']])
        plt.ylabel(PLOT_STRINGS[figure['y']['type']])

        plt.show()
