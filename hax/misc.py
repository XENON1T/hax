from textwrap import dedent


def dataframe_to_wiki(df, float_digits=5, title='Awesome table'):
    """Convert a pandas dataframe to a dokuwiki table (which you can copy-paste onto the XENON wiki)

    :param df: dataframe to convert

    :param float_digits: Round float-ing point values to this number of digits.

    :param title: title of the table.

    """
    table = '^ %s ' % title + '^' * (len(df.columns) - 1) + '^\n'
    table += '^ ' + ' ^ '.join(df.columns) + ' ^\n'
    def do_round(x):
        if isinstance(x, float):
            return round(x, float_digits)
        return x

    for _, row in df.iterrows():
        table += "| " + ' | '.join([str(do_round(x)) for x in row.values.tolist()]) + ' |\n'
    return table


def code_hider():
    """Make a button in the jupyter notebook to hide all code"""
    # Stolen from stackoverflow... forget which question
    # I would really like these buttons for every individual cell.. but I don't know how
    from IPython.display import HTML    # Please keep here, don't want hax to depend on ipython!
    return HTML(dedent('''
                       <script>
                       code_show=true
                       function code_toggle() {
                        if (code_show){
                        $('div.input').hide();
                          } else {
                        $('div.input').show();
                        }
                        code_show = !code_show
                       }
                       $( document ).ready(code_toggle);
                       </script>
                       <form action="javascript:code_toggle()"><input type="submit"
                       value="Show/hide  all code in this notebook"></form>'''))


def draw_box(x, y, **kwargs):
    """Draw rectangle, given x-y boundary tuples"""
    # Arcane syntax of the week: matplotlib's Rectangle...
    import matplotlib
    import matplotlib.pyplot as plt
    plt.gca().add_patch(matplotlib.patches.Rectangle(
        (x[0], y[0]), x[1] - x[0], y[1] - y[0], facecolor='none', **kwargs))
