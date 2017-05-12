from datetime import datetime,date,time
import os.path
import pickle
import pandas as pd
import visdom
import numpy as np

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
    def __init__(self):
        self.svar={}
        self.dvar=[]
        self.t=-1
        self.scopes=[]
        self.file=None
        self.vis=None

    def add_static_value(self,key,value):
        self.svar[key]=value

    def new_iteration(self):
        self.t=self.t+1
        self.dvar.append({})
        self.scopes = []

    def push_scope(self, name):
        self.scopes.append(name)

    def pop_scope(self):
        return self.scopes.pop()

    def _get_dtable(self,scope,t):
        tt=self.dvar[t]
        for s in scope:
            tt=tt[s]
        return tt

    def add_dynamic_value(self,key,value):
        tt=self.dvar[self.t]
        for s in self.scopes:
            if (not s in tt):
                tt[s]={}
            tt=tt[s]

        tt[key]=value

    def get_last_dynamic_value(self,key):
        key=".".join(self.scopes)+key
        return self.dvar[self.t][key]

    def get_column(self,key):
        c=[]
        for d in self.dvar:
            c.append(d[key])
        return c

    def print_static(self):
        print("===== STATIC VARIABLES =========")
        for i in self.svar:
            print(str(i)+" = "+str(self.svar[i]))

    def _generate_columns_names(self):
        columns={}
        scope=[]
        for t in range(self.t):
            tt=self.dvar[t]
            cc=self._generate_columns_names_from_dict(tt,scope)
            for kk in cc.keys():
                columns[kk]=1
        return columns

    def _generate_columns_names_from_dict(self,d,scope):
        columns={}
        for k in d.keys():
            if (isinstance(d[k],dict)):
                scope.append(k)
                cc=self._generate_columns_names_from_dict(d[k],scope)
                for kk in cc.keys():
                    columns[kk]=1
                scope.pop()
            else:
                columns[".".join(scope)+"."+k]=1

        return columns

    def get_scoped_value(self,t,name):
        scope=name.split(".")
        tt=self.dvar[t]
        for s in scope:
            if (not s in tt):
                return None
            tt=tt[s]
        return tt

    def save_file(self,filename=None,directory=None):
        if (directory is None):
            directory="logs"

        if (filename is None):
            filename=str(datetime.now()).replace(" ","_")+".log"
            while(os.path.isfile(directory+"/"+filename)):
                filename = str(datetime.now()).replace(" ", "_")+".log"
        pickle.dump( self, open( directory+"/"+filename, "wb" ) )

    def get_static_values(self):
        return self.svar

    def to_array(self):
        '''
        Transforms the dynamic values to an array
        '''
        names = self._generate_columns_names()
        names["_iteration"] = 1

        retour = []
        cn = []
        for l in names:
            cn.append(l)
        retour.append(cn)

        for t in range(len(self.dvar)):
            cn = []
            for l in names:
                if (l == "_iteration"):
                    cn.append(t)
                else:
                    v = self.get_scoped_value(t, l)
                    cn.append(v)
            retour.append(cn)
        return retour

    def plot_line(self,column_names,win=None,opts={}):
        if (len(self.dvar)<=1):
            return None

        if (self.vis is None):
            self.vis=visdom.Visdom()

        r=[]
        X=[]
        for t in range(len(self.dvar)):
            rr=[]
            for c in column_names:
                rr.append(self.get_scoped_value(t,c))
            r.append(rr)
            X.append(t)

        print(np.array(r).shape)
        print(column_names)
        print(np.array(X).ndim)
        opts_={}
        opts_["legend"]=column_names
        for k in opts:
            opts_[k]=opts[k]
        print(opts_)
        return self.vis.line(X=np.array(X),Y=np.array(r),opts=opts_,win=win)
        #options={"legend":column_names}


    def to_extended_array(self):
        '''
        Transforms the dynamic values to an array
        '''
        names = self._generate_columns_names()
        names["_iteration"] = 1

        for k in self.svar:
            names["_s_"+k]=1

        retour = []
        cn = []
        for l in names:
            cn.append(l)
        retour.append(cn)

        for t in range(len(self.dvar)):
            cn=[]
            for l in names:
                if (l.startswith('_s_')):
                    cn.append(self.svar[l[3:]])
                elif (l == "_iteration"):
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


def logs_to_dataframe(filenames):
    print("Loading %d files and building Dataframe" % len(filenames))
    arrays=[]
    for f in filenames:
        log=pickle.load(open(f,"rb"))
        arrays.append(log.to_extended_array())

    #Building the set of all columns + index per log
    indexes=[]
    all_columns={}
    for i in range(len(arrays)):
        index={}
        columns_names=arrays[i][0]
        for j in range(len(columns_names)):
            index[columns_names[j]]=j
            all_columns[columns_names[j]]=1

        indexes.append(index)

    retour=[]
    all_names=["_log_idx","_log_file"]
    for a in all_columns:
        all_names.append(a)

    for i in range(len(arrays)):
        arr=arrays[i]
        filename=filenames[i]

        for rt in range(len(arr)-1):
            t=rt+1
            line=arr[t]

            new_line=[]
            for idx_c in range(len(all_names)):
                new_line.append(None)
            for idx_c in range(len(all_names)):
                column_name=all_names[idx_c]

                if (column_name == "_log_file"):
                    new_line[idx_c] = filename
                elif (column_name == "_log_idx"):
                    new_line[idx_c] = i
                elif (column_name in indexes[i]):
                    idx = indexes[i][column_name]
                    new_line[idx_c] = arr[t][idx]

            retour.append(new_line)

    return pd.DataFrame(data=retour,columns=all_names)
