#!/usr/bin/env python
import re, sys, argparse

# CONSTANTS

REGEX_COMMENTS = r"(\/\/[^\n]*)|(#[^\n]*)|(\/\*[^\/]*\*\/)"
REGEX_FUNCTIONS = r"(\w+)\("
REGEX_VARS = r"\$(\w+)"
TOLERANCE = 10.0
MAX_VAR_LEN = 255
RESERVED_VARS = [ '_GET', '_POST', '_COOKIE', '_SERVER', '_REQUEST', 'GLOBALS', '_FILES', '_ENV', '_SESSION', '__', '_' ]

#@FIXME I don't like globals (who does?)
global f_az, f_AZ, f_09, f_other
f_az = {}
f_AZ = {}
f_09 = {}
f_other = {}

# Summarize
global RANGES
RANGES = [ 
    (1, 4), 
    (4, 7), 
    (7, 9), 
    (9, 13), 
    (13, 16) 
    ]
#RANGES = [ (1, 4), (4, 8), (8, 12), (12, 100) ]

# http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
class bcolors:
    HEADER = '\033[01;36m'
    SUBHEADER = '\033[01;35m'
    WHITE = '\033[01;37m'
    OKBLUE = '\033[34m'
    OKGREEN = '\033[01;32m'
    WARNING = '\033[01;33m'
    FAIL = '\033[01;31m'
    ENDC = '\033[0m'    

def colorize_float(text, color):

    return colorize("%0.2f%%" % text, color)

def colorize(text, color):

    if color is None:
        return text

    return "%s%s%s" % (color, text, bcolors.ENDC)


def strip_comments(code):

    comments_pattern = re.compile( REGEX_COMMENTS, re.M|re.I|re.S)
    return comments_pattern.sub('', code)

def parse_file(filename, parse_functions=True):
    
    data = None

    with open (filename, "r") as source_file:
        data=source_file.read()

    clean_code = strip_comments(data)

    code_vars = re.findall( REGEX_VARS, clean_code, re.M|re.I|re.S)

    if parse_functions:
        code_functions = re.findall( REGEX_FUNCTIONS, clean_code, re.M|re.I|re.S)
        featureset = set(code_vars+code_functions)
    else:
        featureset = set(code_vars)

    var_features = {}

    for varname in featureset:

        if len(varname) < 1:
            continue

        if varname in RESERVED_VARS:
            continue

        features = {
            'az': re.findall(r"([a-z])", varname, re.M|re.S),
            'AZ': re.findall(r"([A-Z])", varname, re.M|re.S),
            '09': re.findall(r"([0-9])", varname, re.M|re.S),
            'other': re.findall(r"([^A-Za-z0-9])", varname, re.M|re.S),
        }
        var_features[varname] = features

    return var_features

def getrange(nr):
    nr = int(nr)
    for r in RANGES:
        if nr in range(r[0], r[1]):
            return r

    return False

def calculate_stats(var_features):

    print
    print colorize("Calculating statistics from features from candidates' names", bcolors.HEADER)

    var_stats = {}
    for varname, varfeatures in var_features.iteritems():
        varlen = len(varname)
        _az = float(len(varfeatures['az']))/varlen*100
        _AZ = float(len(varfeatures['AZ']))/varlen*100
        _09 = float(len(varfeatures['09']))/varlen*100
        _other = float(len(varfeatures['other']))/varlen*10

        rangename = "%d-%d" % getrange(varlen)

        global f_az, f_AZ, f_09, f_other
        f_az[rangename] += _az
        f_AZ[rangename] += _AZ
        f_09[rangename] += _09
        f_other[rangename] += _other 

        var_stats[varname] = {
            'az': _az,
            'AZ': _AZ,
            '09': _09,
            'other': _other,
        }

    return var_stats

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """

    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib.spines import Spine
        from matplotlib.projections.polar import PolarAxes
        from matplotlib.projections import register_projection    
    except:
        sys.sderr.write("matplotlib/numpy not found")
        sys.exit(1)

    # calculate evenly-spaced axis angles
    theta = 2*np.pi * np.linspace(0, 1-1./num_vars, num_vars)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(theta * 180/np.pi, labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def plot_graph(var_stats, var_clean, var_evil):

    try:
        import matplotlib.pyplot as plt       
    except:
        sys.sderr.write("matploitlib not found")
        sys.exit(1)

    N = 4 # az AZ 09 other
    theta = radar_factory(N, frame='circle')
    spoke_labels = ["[a-z]", "[A-Z]", "[0-9]", "[other]"]

    fig = plt.figure(figsize=(15, 8))
    fig.subplots_adjust(wspace=0.20, hspace=0.20, top=1.00, bottom=0.00, left=0.05, right=0.93)

    """
    # Overall
    ax = fig.add_subplot(2, 2, 3, projection='radar')
    plt.rgrids([20, 40, 60, 80])

    ax.set_title("Overall", weight='bold', size='medium', position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center')

    for varname, varfeatures in var_stats.iteritems():
        d = [ varfeatures['az'], varfeatures['AZ'], varfeatures['09'], varfeatures['other'] ]
        color = 'green' if varname in var_clean else 'red'
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25)
    ax.set_varlabels(spoke_labels)
    """

    # Clean
    ax = fig.add_subplot(1, 2, 1, projection='radar')
    plt.rgrids([20, 40, 60, 80])

    ax.set_title("Clean", weight='bold', size='medium', position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center')

    for varname, varfeatures in var_clean.iteritems():
        d = [ varfeatures['az'], varfeatures['AZ'], varfeatures['09'], varfeatures['other'] ]
        ax.plot(theta, d, color='green')
        ax.fill(theta, d, facecolor='green', alpha=0.25)
    ax.set_varlabels(spoke_labels)

    # Evil
    ax = fig.add_subplot(1, 2, 2, projection='radar')
    plt.rgrids([20, 40, 60, 80])

    ax.set_title("Evil", weight='bold', size='medium', position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center')

    for varname, varfeatures in var_evil.iteritems():
        d = [ varfeatures['az'], varfeatures['AZ'], varfeatures['09'], varfeatures['other'] ]
        ax.plot(theta, d, color='red')
        ax.fill(theta, d, facecolor='red', alpha=0.25)
    ax.set_varlabels(spoke_labels)

    plt.figtext(0.5, 0.965, 'Variable/Function name features distribution',
                ha='center', color='black', weight='bold', size='large')
    plt.figtext(0.5, 0.935, 'PRICK v0.1',
                ha='center', color='black', weight='normal', size='medium')

    plt.show()

def print_banner():

    print colorize("""
  ::::::::::. :::::::..   :::  .,-:::::  :::  .   
  `;;;```.;;;;;;;``;;;;  ;;;,;;;'````'  ;;; .;;,.
  `]]nnn]]'  [[[,/[[['  [[[[[[         [[[[[/'  
  $$$""     $$$$$$c    $$$$$$        _$$$$,    
  888o      888b "88bo,888`88bo,__,o,"888"88o, 
  YMMMb     MMMM   "W" MMM  "YUMMMMMP"MMM "MMP"
  """, bcolors.WARNING)
    print colorize("  PHP Resource Information Conformity checKer", bcolors.WHITE)
    print "  by Jan Seidl <jseidl at wroot dot org>"
    print

def init_ranges(ranges):

    newranges = []

    global RANGES
    if len(ranges) == 0:
        ranges = RANGES

    lastnum = ranges[-1:][0][1]

    ranges.append((lastnum, MAX_VAR_LEN))

    global f_az, f_AZ, f_09, f_other
    for r in ranges:
        rangename = "%d-%d" % tuple(r)
        f_az[rangename] = 0.0
        f_AZ[rangename] = 0.0
        f_09[rangename] = 0.0
        f_other[rangename] = 0.0
        newranges.append(tuple(r))

    RANGES = newranges
    return newranges

def main():

    parser = argparse.ArgumentParser(
        description = "Check PHP variables and functions names for anomalies",
    )
    parser.add_argument('filename', nargs = '+', help = 'input PHP files for analisys', metavar='FILE')
    parser.add_argument('-t', '--tolerance', type=float, help = 'tolerance offset for statistical analisys (float)', default=10.0)
    parser.add_argument('-p', '--plot', help = 'plot features graph (requires matplotlib and numpy)', action='store_true')
    parser.add_argument('-e', '--evil-only', help = 'display only entries marked as evil', action='store_true')
    parser.add_argument('-u', '--ignore-all-uppercase', help = 'filter out entries of all uppercased vars', action='store_true')
    parser.add_argument('-l', '--ignore-all-lowercase', help = 'filter out entries of all lowercased vars', action='store_true')
    parser.add_argument('-f', '--parse-functions', help = 'parse also function names instead of variable names only', action='store_true')
    parser.add_argument('-r', '--range', help = 'add a range of name size to the classificator (END is non inclusive)', nargs=2, metavar=('START', 'END'), type=int, action="append", default=[])
    args = parser.parse_args()

    TOLERANCE = args.tolerance

    fileref = {}
    featureset = {}
    var_data = []

    print_banner()

    init_ranges(args.range)

    print colorize("Analyzing files:", bcolors.HEADER)
    print
    for filename in args.filename:
        print "  - %s" % filename
        parsed = parse_file(filename, args.parse_functions)
        var_data.append(parsed) # parse_file returns a dict with "variable_name": { "az": X, "AZ": Y, "09": Z, "other": W }
        # Create var->file correlation
        for vk in parsed.keys():
            if vk not in fileref:
                fileref[vk] = []
                
            fileref[vk].append(filename)

    for varset in var_data: # list of dicts
        for varname, varfeatures in varset.iteritems(): # dict
            if varname not in featureset:
                featureset[varname] = varfeatures

    var_stats = calculate_stats(featureset)
    global f_az, f_AZ, f_09, f_other

    nrvars = {}
    total_candidates = len(featureset)
    for varname in featureset.keys():
        rangename = "%d-%d" % getrange(len(varname))
        if rangename in nrvars:
            nrvars[rangename] += 1
        else:
            nrvars[rangename] = 1

    print
    print colorize("  Average: (over %d total unique candidates)" % total_candidates, bcolors.SUBHEADER)

    avg_az = {}
    avg_AZ = {}
    avg_09 = {}
    avg_other = {}

    for r in RANGES:
        rangename = "%d-%d" % r
        avg_az[rangename] = f_az[rangename]/nrvars[rangename]
        avg_AZ[rangename] = f_AZ[rangename]/nrvars[rangename]
        avg_09[rangename] = f_09[rangename]/nrvars[rangename]
        avg_other[rangename] = f_other[rangename]/nrvars[rangename]
        print "  %-*s %-*s %-*s %-*s %-*s" % (20, colorize("[%s]" % rangename, bcolors.WHITE), 40, colorize("[a-z]", bcolors.SUBHEADER) + ' ' + colorize_float(avg_az[rangename], bcolors.WHITE), 40, colorize("[A-Z]", bcolors.SUBHEADER) + ' ' + colorize_float(avg_AZ[rangename], bcolors.WHITE), 40, colorize("[0-9]", bcolors.SUBHEADER) + ' ' + colorize_float(avg_09[rangename], bcolors.WHITE), 40, colorize("[other]", bcolors.SUBHEADER) + ' ' + colorize_float(avg_other[rangename], bcolors.WHITE))

    print
    print colorize("Analyzing frequencies with a %0.2f%% tolerance ratio" % TOLERANCE, bcolors.HEADER)
    print

    var_clean = {}
    var_evil = {}

    print "  %-*s %-*s %-*s %-*s %-*s %-*s" % (47, colorize("CANDIDATE NAME", bcolors.SUBHEADER), 22, colorize("[a-z]", bcolors.SUBHEADER), 22, colorize("[A-Z]", bcolors.SUBHEADER), 22, colorize("[0-9]", bcolors.SUBHEADER), 22, colorize("[OTHER]", bcolors.SUBHEADER), 47, colorize("[FILES]", bcolors.SUBHEADER))
    for varname, varfeatures in var_stats.iteritems():
        c_varname, c_az, c_AZ, c_09, c_other = (None, None, None, None, None)

        varlen = len(varname)
        rangename = "%d-%d" % getrange(varlen)

        if not float(avg_az[rangename]-TOLERANCE) < varfeatures['az'] < float(avg_az[rangename]+TOLERANCE):
            c_az = bcolors.FAIL
        if not float(avg_AZ[rangename]-TOLERANCE) < varfeatures['AZ'] < float(avg_AZ[rangename]+TOLERANCE):
            c_AZ = bcolors.FAIL
        if not float(avg_09[rangename]-TOLERANCE) < varfeatures['09'] < float(avg_09[rangename]+TOLERANCE):
            c_09 = bcolors.FAIL
        if not float(avg_other[rangename]-TOLERANCE) < varfeatures['other'] < float(avg_other[rangename]+TOLERANCE):
            c_other = bcolors.FAIL

        is_evil = False

        if args.ignore_all_lowercase and (varfeatures['az'] >= 90 and varfeatures['AZ'] == 0 and varfeatures['09'] == 0 and varfeatures['other'] <= 1):
            continue

        if args.ignore_all_uppercase and (varfeatures['az'] == 0 and varfeatures['AZ'] >= 90 and varfeatures['09'] == 0 and varfeatures['other'] <= 1):
            continue

        if not all(x is None for x in (c_az, c_AZ, c_09, c_other)):
            is_evil = True
            c_varname = bcolors.WARNING
            var_evil[varname] = varfeatures
        else:
            var_clean[varname] = varfeatures

        if not is_evil and args.evil_only:
            continue

        # Tabwidths for compensating terminal color char sequences
        tw_varname = 47 if c_varname is not None else 35
        tw_az = 22 if c_az is not None else 10
        tw_AZ = 22 if c_AZ is not None else 10
        tw_09 = 22 if c_09 is not None else 10
        tw_other = 22 if c_other is not None else 10

        fref = ', '.join(fileref[varname])

        print "  %-*s %-*s %-*s %-*s %-*s %s" % (tw_varname, colorize(varname, c_varname), tw_az, colorize_float(varfeatures['az'], c_az), tw_AZ, colorize_float(varfeatures['AZ'], c_AZ), tw_09, colorize_float(varfeatures['09'], c_09), tw_other, colorize_float(varfeatures['other'], c_other), fref)

    print
    if len(var_evil) == 0:
        print colorize("No evil candidates were detected.", bcolors.OKGREEN)
    else:
        print colorize("Total %d evil candidates found." % len(var_evil), bcolors.FAIL)
    print

    if args.plot:
        plot_graph(var_stats, var_clean, var_evil)

if __name__ == '__main__':
    main()
