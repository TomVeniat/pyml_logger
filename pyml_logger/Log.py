import os.path
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import visdom


class Log:
    '''
    A log is composed of:
        - static key,value pairs (for example: hyper parameters of the experiment)
        - set of key,value pairs at each iteration

    Typical use is:
        log.add_static_value("learning_rate",0.01)

        for t in range(T):
            perf=evaluate_model()
            log.new_iteration()
            log.add_dynamic_value("perf",perf)
            log.add_dynamic_value("iteration",t)
    '''

    ITERATION_KEY = '_iteration'
    SCOPE_SEP = '.'

    def __init__(self):
        self.s_var = {}
        self.d_var = []
        self.t = -1
        self.scopes = []
        self.file = None
        self.vis = None

    def new_iteration(self):
        self.t += 1
        self.d_var.append({})
        self.add_dynamic_value(self.ITERATION_KEY, self.t)
        self.scopes = []

    def push_scope(self, name):
        self.scopes.append(name)

    def push_scopes(self, names):
        '''
        Push several scopes.
        :param names: Array or string of :attr:`SCOPE_SEP` separated scope names.
            ex: 'scope_1.scope2.[...].scope_n'
                ['scope_1', 'scope2', [...], 'scope_n']
        :return: the depth of added scopes
        '''
        if names is None:
            return 0
        if isinstance(names, str):
            names = names.split(self.SCOPE_SEP)
        for s in names:
            self.push_scope(s)
        return len(names)

    def pop_scope(self):
        return self.scopes.pop()

    def pop_scopes(self, n):
        for _ in range(n):
            self.pop_scope()

    def _get_dtable(self, scope, t):
        '''
        Returns the dynamic values registered at time :t: within the scope :scope:
        '''
        tt = self.d_var[t]
        for s in scope:
            tt = tt[s]
        return tt

    def add_static_value(self, key, value):
        self.s_var[key] = value

    def add_static_values(self, **kwargs):
        for k, v in kwargs.items():
            self.add_static_value(k, v)

    def add_dynamic_value(self, key, value):
        tt = self.d_var[self.t]
        for s in self.scopes:
            if s not in tt:
                tt[s] = {}
            tt = tt[s]
        tt[key] = value

    def add_dynamic_values(self, scope_=None, **kwargs):
        '''
        Add all dynamic values given in kwargs (expect 'scope_') to 
        :param scope_: Scope in which the values will be added. See :func:`~pyml_logger.Log.push_scopes` for format.
        :param kwargs: Each kwarg will be added to the logger under the given scope (can't have a 'scope_' key for now).
        '''
        n_scopes = self.push_scopes(scope_)
        for k, v in kwargs.items():
            self.add_dynamic_value(k, v)
        self.pop_scopes(n_scopes)

    def get_last_dynamic_value(self, key):
        '''
        :param key: The key of the target element.
        :return: The value from current iteration associated with the given key found in the deepest level of 
        the current scope.
        
        Todo: Precise behaviour in some case, for example:
            * Only search in current iteration ? 
            * What if a more recent value is present in another scope ?
        '''
        values = self.d_var[self.t]
        res = None
        if key in values:
            res = values[key]
        for s in self.scopes:
            if s not in values:
                break
            values = values[s]
            if key in values:
                res = values[key]
        return res

    def get_column(self, key):
        c = []
        for d in self.d_var:
            c.append(d[key])
        return c

    def print_static(self):
        print("===== STATIC VARIABLES =========")
        for i in self.s_var:
            print(str(i) + " = " + str(self.s_var[i]))

    def _generate_columns_names(self):
        columns = set()
        for t in range(self.t + 1):
            tt = self.d_var[t]
            cc = self._generate_columns_names_from_dict(tt)
            columns.update(cc)
        return list(columns)

    def _generate_columns_names_from_dict(self, d, scope=None):
        if scope is None:
            scope = []
        columns = set()
        for k in d.keys():
            if isinstance(d[k], dict):
                scope.append(k)
                cc = self._generate_columns_names_from_dict(d[k], scope)
                columns.update(cc)
                scope.pop()
            else:
                columns.add(".".join(scope + [k]))
        return columns

    def get_scoped_value(self, t, name):
        scope = name.split(".")
        tt = self.d_var[t]
        for s in scope:
            if s not in tt:
                return None
            tt = tt[s]
        return tt

    def save_file(self, filename=None, directory=None):
        if (directory is None):
            directory = "logs"

        if (filename is None):
            filename = str(datetime.now()).replace(" ", "_") + ".log"
            while (os.path.isfile(directory + "/" + filename)):
                filename = str(datetime.now()).replace(" ", "_") + ".log"
        pickle.dump(self, open(directory + "/" + filename, "wb"))

    def get_static_values(self):
        return self.s_var

    def to_array(self):
        '''
        Transforms the dynamic values to an array.
        The first 
        '''
        names = self._generate_columns_names()
        res = [names]

        for t in range(len(self.d_var)):
            vals = []
            for name in names:
                v = self.get_scoped_value(t, name)
                vals.append(v)
            res.append(vals)
        return res

    def plot_line(self, columns, x_name=None, win=None, opts=None):
        if opts is None:
            opts = {}
        if len(self.d_var) == 0:
            return None

        if type(columns) is not list:
            columns = [columns]

        if x_name is None:
            x_name = self.ITERATION_KEY

        if self.vis is None:
            self.vis = visdom.Visdom()

        x = []
        values = []

        for t in range(self.t+1):
            cur_values = []
            for key in columns:
                cur_values.append(self.get_scoped_value(t, key))

            if not None in cur_values:
                values.append(cur_values)
                x.append(self.get_scoped_value(t, x_name))

        opts_ = {"legend": columns}
        opts_.update(opts)

        x = np.array(x)
        y = np.array(values)
        if len(columns) == 1:
            y = y.flatten()


        return self.vis.line(X=x, Y=y, opts=opts_, win=win)

    def to_extended_array(self):
        '''
        Transforms the dynamic and static values to an array
        '''
        names = self._generate_columns_names()
        names.insert(0, "_iteration")

        for k in self.s_var:
            names.append("_s_" + k)

        retour = []
        cn = []
        for l in names:
            cn.append(l)
        retour.append(cn)

        for t in range(len(self.d_var)):
            cn = []
            for l in names:
                if l.startswith('_s_'):
                    cn.append(self.s_var[l[3:]])
                elif l == "_iteration":
                    cn.append(t)
                else:
                    v = self.get_scoped_value(t, l)
                    cn.append(v)
            retour.append(cn)
        return retour

    def to_dataframe(self):
        a = self.to_array()
        return pd.DataFrame(data=a[1:], columns=a[0])

    def to_extended_dataframe(self):
        a = self.to_extended_array()
        return pd.DataFrame(data=a[1:], columns=a[0])

    def get_state(self):
            return {
                't': self.t,
                'd_var': self.d_var
            }

    def set_state(self, state):
        self.t = state['t']
        assert len(self.d_var) == 0
        for step in state['d_var']:
            self.d_var.append(step)

def logs_to_dataframe(filenames):
    print("Loading %d files and building Dataframe" % len(filenames))
    arrays = []
    for f in filenames:
        log = pickle.load(open(f, "rb"))
        arrays.append(log.to_extended_array())

    # Building the set of all columns + index per log
    indexes = []
    all_columns = {}
    for i in range(len(arrays)):
        index = {}
        columns_names = arrays[i][0]
        for j in range(len(columns_names)):
            index[columns_names[j]] = j
            all_columns[columns_names[j]] = 1

        indexes.append(index)

    retour = []
    all_names = ["_log_idx", "_log_file"]
    for a in all_columns:
        all_names.append(a)

    for i in range(len(arrays)):
        arr = arrays[i]
        filename = filenames[i]

        for rt in range(len(arr) - 1):
            t = rt + 1
            line = arr[t]

            new_line = []
            for idx_c in range(len(all_names)):
                new_line.append(None)
            for idx_c in range(len(all_names)):
                column_name = all_names[idx_c]

                if (column_name == "_log_file"):
                    new_line[idx_c] = filename
                elif (column_name == "_log_idx"):
                    new_line[idx_c] = i
                elif (column_name in indexes[i]):
                    idx = indexes[i][column_name]
                    new_line[idx_c] = arr[t][idx]

            retour.append(new_line)

    return pd.DataFrame(data=retour, columns=all_names)
