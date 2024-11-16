import numpy as np
import matplotlib.pyplot as plt

# 提供的数据
data = {
    'base': {'b': {'mean': 0.585278, 'std': 0.071789}, 'g': {'mean': 0.221526, 'std': 0.019394}},
    'r1': {'b': {'mean': 0.593472, 'std': 0.026841}, 'g': {'mean': 0.241701, 'std': 0.020894}},
    'r3': {'b': {'mean': 0.545773, 'std': 0.056889}, 'g': {'mean': 0.250462, 'std': 0.027237}},
    'r5': {'b': {'mean': 0.550408, 'std': 0.041426}, 'g': {'mean': 0.264468, 'std': 0.009836}},
}
data_tar_1000 = {
    'base': {
        'b': {'qini': {'mean': 0.585278, 'std': 0.071789}, 'mape': {'mean': 0.408304, 'std': 0.077749}, 'copc': {'mean': 1.731624, 'std': 0.231094}},
        'g': {'qini': {'mean': 0.221526, 'std': 0.019394}, 'mape': {'mean': 0.272826, 'std': 0.088646}, 'copc': {'mean': 0.968651, 'std': 0.146562}}
    },
    'r1': {
        'b': {'qini': {'mean': 0.593472, 'std': 0.026841}, 'mape': {'mean': 0.482603, 'std': 0.083064}, 'copc': {'mean': 1.958466, 'std': 0.288041}},
        'g': {'qini': {'mean': 0.241701, 'std': 0.020894}, 'mape': {'mean': 0.202234, 'std': 0.082827}, 'copc': {'mean': 1.21025, 'std': 0.20401}}
    },
    'r1 random': {
        'b': {'qini': {'mean': 0.581956, 'std': 0.036438}, 'mape': {'mean': 0.364761, 'std': 0.098921}, 'copc': {'mean': 1.604467, 'std': 0.294105}},
        'g': {'qini': {'mean': 0.235568, 'std': 0.018646}, 'mape': {'mean': 0.200485, 'std': 0.075857}, 'copc': {'mean': 0.967816, 'std': 0.146500}}
    },
    'r3': {
        'b': {'qini': {'mean': 0.545773, 'std': 0.056889}, 'mape': {'mean': 0.328359, 'std': 0.079258}, 'copc': {'mean': 1.516814, 'std': 0.206710}},
        'g': {'qini': {'mean': 0.250462, 'std': 0.027237}, 'mape': {'mean': 0.218816, 'std': 0.159826}, 'copc': {'mean': 0.927264, 'std': 0.149671}}
    },
    'r3 random': {
        'b': {'qini': {'mean': 0.516962, 'std': 0.090401}, 'mape': {'mean': 0.380017, 'std': 0.109324}, 'copc': {'mean': 1.601168, 'std': 0.368659}},
        'g': {'qini': {'mean': 0.221766, 'std': 0.024079}, 'mape': {'mean': 0.259213, 'std': 0.124832}, 'copc': {'mean': 0.972946, 'std': 0.226222}}
    },
    'r5': {
        'b': {'qini': {'mean': 0.550408, 'std': 0.041426}, 'mape': {'mean': 0.326225, 'std': 0.065351}, 'copc': {'mean': 1.540597, 'std': 0.176460}},
        'g': {'qini': {'mean': 0.264468, 'std': 0.009836}, 'mape': {'mean': 0.116791, 'std': 0.055483}, 'copc': {'mean': 0.976424, 'std': 0.108368}}
    },
    'r5 random': {
        'b': {'qini': {'mean': 0.583626, 'std': 0.040949}, 'mape': {'mean': 0.291009, 'std': 0.083829}, 'copc': {'mean': 1.437020, 'std': 0.176713}},
        'g': {'qini': {'mean': 0.25503, 'std': 0.01056}, 'mape': {'mean': 0.267097, 'std': 0.098337}, 'copc': {'mean': 0.896270, 'std': 0.100177}}
    }
}
data_tar_5000 = {
    'base': {
        'b': {'qini': {'mean': 0.364934, 'std': 0.027073}, 'mape': {'mean': 0.289804, 'std': 0.054896}, 'copc': {'mean': 1.449705, 'std': 0.121844}},
        'g': {'qini': {'mean': 0.258717, 'std': 0.007219}, 'mape': {'mean': 0.090626, 'std': 0.037371}, 'copc': {'mean': 1.047265, 'std': 0.089866}}
    },
    'r1': {
        'b': {'qini': {'mean': 0.414962, 'std': 0.026023}, 'mape': {'mean': 0.257179, 'std': 0.078897}, 'copc': {'mean': 1.409029, 'std': 0.187377}},
        'g': {'qini': {'mean': 0.298725, 'std': 0.011096}, 'mape': {'mean': 0.094686, 'std': 0.064934}, 'copc': {'mean': 1.022646, 'std': 0.131487}}
    },
    'r1 random': {
        'b': {'qini': {'mean': 0.428192, 'std': 0.046773}, 'mape': {'mean': 0.243781, 'std': 0.052350}, 'copc': {'mean': 1.366094, 'std': 0.112707}},
        'g': {'qini': {'mean': 0.295948, 'std': 0.020191}, 'mape': {'mean': 0.079952, 'std': 0.094255}, 'copc': {'mean': 0.986162, 'std': 0.094369}}
    },
    'r3': {
        'b': {'qini': {'mean': 0.445539, 'std': 0.024719}, 'mape': {'mean': 0.229187, 'std': 0.061040}, 'copc': {'mean': 1.346288, 'std': 0.145500}},
        'g': {'qini': {'mean': 0.322620, 'std': 0.005227}, 'mape': {'mean': 0.078339, 'std': 0.073988}, 'copc': {'mean': 0.981308, 'std': 0.101330}}
    },
    'r3 random': {
        'b': {'qini': {'mean': 0.428478, 'std': 0.025019}, 'mape': {'mean': 0.209986, 'std': 0.067135}, 'copc': {'mean': 1.305447, 'std': 0.133923}},
        'g': {'qini': {'mean': 0.295641, 'std': 0.008286}, 'mape': {'mean': 0.101292, 'std': 0.068800}, 'copc': {'mean': 0.949521, 'std': 0.092315}}
    },
    'r5': {
        'b': {'qini': {'mean': 0.443859, 'std': 0.021715}, 'mape': {'mean': 0.211452, 'std': 0.059247}, 'copc': {'mean': 1.313163, 'std': 0.112674}},
        'g': {'qini': {'mean': 0.328351, 'std': 0.006134}, 'mape': {'mean': 0.077908, 'std': 0.066277}, 'copc': {'mean': 0.961149, 'std': 0.078875}}
    },
    'r5 random': {
        'b': {'qini': {'mean': 0.396066, 'std': 0.020148}, 'mape': {'mean': 0.205680, 'std': 0.073364}, 'copc': {'mean': 1.296917, 'std': 0.138035}},
        'g': {'qini': {'mean': 0.290794, 'std': 0.012216}, 'mape': {'mean': 0.117600, 'std': 0.075929}, 'copc': {'mean': 0.958387, 'std': 0.101349}}
    }
}
data_tar_10k = {
    'base': {
        'b': {'qini': {'mean': 0.345569, 'std': 0.043732}, 'mape': {'mean': 0.183699, 'std': 0.126380}, 'copc': {'mean': 1.261038, 'std': 0.287990}},
        'g': {'qini': {'mean': 0.298961, 'std': 0.023843}, 'mape': {'mean': 0.169907, 'std': 0.101356}, 'copc': {'mean': 1.002825, 'std': 0.209748}}
    },
    'r1': {
        'b': {'qini': {'mean': 0.363865, 'std': 0.025412}, 'mape': {'mean': 0.213849, 'std': 0.084893}, 'copc': {'mean': 1.321259, 'std': 0.180761}},
        'g': {'qini': {'mean': 0.320372, 'std': 0.010827}, 'mape': {'mean': 0.111645, 'std': 0.076512}, 'copc': {'mean': 1.07178, 'std': 0.14945}}
    },
    'r1 random': {
        'b': {'qini': {'mean': 0.363204, 'std': 0.010390}, 'mape': {'mean': 0.121909, 'std': 0.035089}, 'copc': {'mean': 1.130084, 'std': 0.080023}},
        'g': {'qini': {'mean': 0.329081, 'std': 0.006385}, 'mape': {'mean': 0.114127, 'std': 0.065785}, 'copc': {'mean': 0.914436, 'std': 0.059911}}
    },
    'r3': {
        'b': {'qini': {'mean': 0.357017, 'std': 0.010031}, 'mape': {'mean': 0.174490, 'std': 0.045228}, 'copc': {'mean': 1.237386, 'std': 0.093031}},
        'g': {'qini': {'mean': 0.340933, 'std': 0.003816}, 'mape': {'mean': 0.057123, 'std': 0.046194}, 'copc': {'mean': 1.021188, 'std': 0.072791}}
    },
    'r3 random': {
        'b': {'qini': {'mean': 0.373884, 'std': 0.026513}, 'mape': {'mean': 0.189373, 'std': 0.066832}, 'copc': {'mean': 1.240913, 'std': 0.169628}},
        'g': {'qini': {'mean': 0.327047, 'std': 0.010646}, 'mape': {'mean': 0.107738, 'std': 0.088850}, 'copc': {'mean': 0.991935, 'std': 0.124662}}
    },
    'r5': {
        'b': {'qini': {'mean': 0.357173, 'std': 0.009188}, 'mape': {'mean': 0.179063, 'std': 0.041121}, 'copc': {'mean': 1.244751, 'std': 0.088778}},
        'g': {'qini': {'mean': 0.346422, 'std': 0.002137}, 'mape': {'mean': 0.054619, 'std': 0.042798}, 'copc': {'mean': 1.029660, 'std': 0.071658}}
    },
    'r5 random': {
        'b': {'qini': {'mean': 0.378843, 'std': 0.020468}, 'mape': {'mean': 0.154322, 'std': 0.063965}, 'copc': {'mean': 1.187920, 'std': 0.134383}},
        'g': {'qini': {'mean': 0.327543, 'std': 0.007575}, 'mape': {'mean': 0.108379, 'std': 0.073252}, 'copc': {'mean': 0.960755, 'std': 0.104929}}
    }
}

data_cf_1000 = {
    'base': {
        'b': {'qini': {'mean': 0.604759, 'std': 0.023129}, 'mape': {'mean': 0.419478, 'std': 0.015616}, 'copc': {'mean': 1.656856, 'std': 0.099891}},
        'g': {'qini': {'mean': 0.273809, 'std': 0.000953}, 'mape': {'mean': 0.140154, 'std': 0.000643}, 'copc': {'mean': 1.032063, 'std': 0.032108}}
    },
    'r1': {
        'b': {'qini': {'mean': 0.496646, 'std': 0.000000}, 'mape': {'mean': 0.541678, 'std': 0.000000}, 'copc': {'mean': 1.520211, 'std': 2.220446e-16}},
        'g': {'qini': {'mean': 0.292144, 'std': 0.000000}, 'mape': {'mean': 9.173285e-02, 'std': 1.213811e-16}, 'copc': {'mean': 1.108066, 'std': 1.570092e-16}}
    },
    'r1 random': {
        'b': {'qini': {'mean': 0.580272, 'std': 0.000000}, 'mape': {'mean': 3.217943e-01, 'std': 9.614813e-17}, 'copc': {'mean': 1.49796, 'std': 0.00000}},
        'g': {'qini': {'mean': 0.299937, 'std': 0.000000}, 'mape': {'mean': 8.181618e-02, 'std': 5.721958e-17}, 'copc': {'mean': 1.016806, 'std': 2.719480e-16}}
    },
    'r3': {
        'b': {'qini': {'mean': 0.517859, 'std': 0.039563}, 'mape': {'mean': 0.371516, 'std': 0.032221}, 'copc': {'mean': 1.524704, 'std': 0.019950}},
        'g': {'qini': {'mean': 0.315603, 'std': 0.003204}, 'mape': {'mean': 0.102073, 'std': 0.016376}, 'copc': {'mean': 1.117499, 'std': 0.034416}}
    },
    'r3 random': {
        'b': {'qini': {'mean': 0.625581, 'std': 0.005555}, 'mape': {'mean': 0.392155, 'std': 0.005829}, 'copc': {'mean': 1.662562, 'std': 0.007092}},
        'g': {'qini': {'mean': 0.295916, 'std': 0.005485}, 'mape': {'mean': 0.119801, 'std': 0.000505}, 'copc': {'mean': 1.108948, 'std': 0.000070}}
    },
    'r5': {
        'b': {'qini': {'mean': 0.505435, 'std': 0.000000}, 'mape': {'mean': 3.159995e-01, 'std': 6.798700e-17}, 'copc': {'mean': 1.450078, 'std': 2.220446e-16}},
        'g': {'qini': {'mean': 0.31567, 'std': 0.00000}, 'mape': {'mean': 7.954722e-02, 'std': 5.637184e-17}, 'copc': {'mean': 1.091917, 'std': 1.570092e-16}}
    },
    'r5 random': {
        'b': {'qini': {'mean': 0.564120, 'std': 0.010904}, 'mape': {'mean': 0.812512, 'std': 0.336725}, 'copc': {'mean': 1.375708, 'std': 0.059779}},
        'g': {'qini': {'mean': 0.288495, 'std': 0.000802}, 'mape': {'mean': 0.142379, 'std': 0.010884}, 'copc': {'mean': 1.010145, 'std': 0.016437}}
    }
}
data_cf_5000 = {
    'base': {
        'b': {'qini': {'mean': 0.394275, 'std': 0.006997}, 'mape': {'mean': 0.355582, 'std': 0.005672}, 'copc': {'mean': 1.606895, 'std': 0.021892}},
        'g': {'qini': {'mean': 0.303774, 'std': 0.001461}, 'mape': {'mean': 0.114652, 'std': 0.007667}, 'copc': {'mean': 1.131202, 'std': 0.008889}}
    },
    'r1': {
        'b': {'qini': {'mean': 0.406672, 'std': 0.005010}, 'mape': {'mean': 0.296068, 'std': 0.003443}, 'copc': {'mean': 1.456335, 'std': 0.007978}},
        'g': {'qini': {'mean': 0.322795, 'std': 0.000877}, 'mape': {'mean': 0.054018, 'std': 0.001044}, 'copc': {'mean': 1.058913, 'std': 0.001457}}
    },
    'r1 random': {
        'b': {'qini': {'mean': 0.427761, 'std': 0.002993}, 'mape': {'mean': 0.307114, 'std': 0.004784}, 'copc': {'mean': 1.474985, 'std': 0.007641}},
        'g': {'qini': {'mean': 0.322239, 'std': 0.000190}, 'mape': {'mean': 0.062563, 'std': 0.006396}, 'copc': {'mean': 1.069808, 'std': 0.007888}}
    },
    'r3': {
        'b': {'qini': {'mean': 0.411144, 'std': 0.001556}, 'mape': {'mean': 0.288230, 'std': 0.000032}, 'copc': {'mean': 1.442619, 'std': 0.001765}},
        'g': {'qini': {'mean': 0.329595, 'std': 0.000154}, 'mape': {'mean': 0.046809, 'std': 0.003028}, 'copc': {'mean': 1.050047, 'std': 0.003676}}
    },
    'r3 random': {
        'b': {'qini': {'mean': 0.426041, 'std': 0.001185}, 'mape': {'mean': 0.296534, 'std': 0.001859}, 'copc': {'mean': 1.442611, 'std': 0.001065}},
        'g': {'qini': {'mean': 0.325438, 'std': 0.000502}, 'mape': {'mean': 0.070351, 'std': 0.000435}, 'copc': {'mean': 1.055532, 'std': 0.002856}}
    },
    'r5': {
        'b': {'qini': {'mean': 0.419484, 'std': 0.001757}, 'mape': {'mean': 0.288060, 'std': 0.005189}, 'copc': {'mean': 1.443962, 'std': 0.008820}},
        'g': {'qini': {'mean': 0.331629, 'std': 0.000495}, 'mape': {'mean': 0.046379, 'std': 0.003584}, 'copc': {'mean': 1.049894, 'std': 0.004236}}
    },
    'r5 random': {
        'b': {'qini': {'mean': 0.419051, 'std': 0.000000}, 'mape': {'mean': 3.103158e-01, 'std': 1.177569e-16}, 'copc': {'mean': 1.482019, 'std': 2.719480e-16}},
        'g': {'qini': {'mean': 0.319431, 'std': 0.000000}, 'mape': {'mean': 0.072453, 'std': 0.000000}, 'copc': {'mean': 1.065466, 'std': 1.570092e-16}}
    }
}
data_cf_10k = {
    'base': {
        'b': {'qini': {'mean': 0.347767, 'std': 0.003622}, 'mape': {'mean': 0.236346, 'std': 0.008862}, 'copc': {'mean': 1.325453, 'std': 0.017029}},
        'g': {'qini': {'mean': 0.318336, 'std': 0.000161}, 'mape': {'mean': 0.062064, 'std': 0.003230}, 'copc': {'mean': 1.056415, 'std': 0.002137}}
    },
    'r1': {
        'b': {'qini': {'mean': 0.348197, 'std': 0.001393}, 'mape': {'mean': 0.226404, 'std': 0.001915}, 'copc': {'mean': 1.312428, 'std': 0.003231}},
        'g': {'qini': {'mean': 0.329185, 'std': 0.000325}, 'mape': {'mean': 0.054764, 'std': 0.001528}, 'copc': {'mean': 1.060025, 'std': 0.001801}}
    },
    'r1 random': {
        'b': {'qini': {'mean': 0.343058, 'std': 0.006528}, 'mape': {'mean': 0.206804, 'std': 0.003601}, 'copc': {'mean': 1.281002, 'std': 0.006547}},
        'g': {'qini': {'mean': 0.328834, 'std': 0.000597}, 'mape': {'mean': 0.050016, 'std': 0.002370}, 'copc': {'mean': 1.041487, 'std': 0.002130}}
    },
    'r3': {
        'b': {'qini': {'mean': 0.351405, 'std': 0.009109}, 'mape': {'mean': 0.226274, 'std': 0.004898}, 'copc': {'mean': 1.314783, 'std': 0.010159}},
        'g': {'qini': {'mean': 0.333713, 'std': 0.000235}, 'mape': {'mean': 0.052096, 'std': 0.003688}, 'copc': {'mean': 1.056632, 'std': 0.004193}}
    },
    'r3 random': {
        'b': {'qini': {'mean': 0.346498, 'std': 0.000000}, 'mape': {'mean': 2.208097e-01, 'std': 6.509259e-17}, 'copc': {'mean': 1.299368, 'std': 0.000000}},
        'g': {'qini': {'mean': 0.328097, 'std': 0.000000}, 'mape': {'mean': 5.333844e-02, 'std': 8.498375e-18}, 'copc': {'mean': 1.053268, 'std': 0.000000}}
    },
    'r5': {
        'b': {'qini': {'mean': 0.363317, 'std': 0.002236}, 'mape': {'mean': 0.220540, 'std': 0.002861}, 'copc': {'mean': 1.306061, 'std': 0.004370}},
        'g': {'qini': {'mean': 0.336949, 'std': 0.000192}, 'mape': {'mean': 0.048691, 'std': 0.001339}, 'copc': {'mean': 1.052618, 'std': 0.001534}}
    },
    'r5 random': {
        'b': {'qini': {'mean': 3.530961e-01, 'std': 6.798700e-17}, 'mape': {'mean': 2.215790e-01, 'std': 2.246316e-16}, 'copc': {'mean': 1.299158e+00, 'std': 3.845925e-16}},
        'g': {'qini': {'mean': 0.327549, 'std': 0.000000}, 'mape': {'mean': 0.064635, 'std': 0.000000}, 'copc': {'mean': 1.054662e+00, 'std': 2.719480e-16}}
    }
}

data = {
    'base': {'b': {'mean': 0.364934, 'std': 0.027073}, 'g': {'mean': 0.258717, 'std': 0.007219}},
    'r1': {'b': {'mean': 0.414962, 'std': 0.026023}, 'g': {'mean': 0.298725, 'std': 0.011096}},
    'r3': {'b': {'mean': 0.445539, 'std': 0.024719}, 'g': {'mean': 0.322620, 'std': 0.005227}},
    'r5': {'b': {'mean': 0.443859, 'std': 0.021715}, 'g': {'mean': 0.328351, 'std': 0.006134}},
}

data = {
    'base': {'b': {'mean': 0.345569, 'std': 0.043732}, 'g': {'mean': 0.298961, 'std': 0.023843}},
    'r1': {'b': {'mean': 0.363865, 'std': 0.025412}, 'g': {'mean': 0.320372, 'std': 0.010827}},
    'r3': {'b': {'mean': 0.357017, 'std': 0.010031}, 'g': {'mean': 0.340933, 'std': 0.003816}},
    'r5': {'b': {'mean': 0.357173, 'std': 0.009188}, 'g': {'mean': 0.346422, 'std': 0.002137}},
}
###mape
data = {
    'base': {'b': {'mean': 0.408304, 'std': 0.077749}, 'g': {'mean': 0.272826, 'std': 0.088646}},
    'r1': {'b': {'mean': 0.482603, 'std': 0.083064}, 'g': {'mean': 0.202234, 'std': 0.082827}},
    'r3': {'b': {'mean': 0.328359, 'std': 0.079258}, 'g': {'mean': 0.218816, 'std': 0.159826}},
    'r5': {'b': {'mean': 0.326225, 'std': 0.065351}, 'g': {'mean': 0.116791, 'std': 0.055483}},
}
# 提取数据

x_labels = ['TARNet', 'TARNet(+1)', 'TARNet(+3)', 'TARNet(+5)']
data = data_tar_1000
number = 1000
name = 'copc'
b_means = [data['base']['b'][name]['mean'], data['r1']['b'][name]['mean'], data['r3']['b'][name]['mean'], data['r5']['b'][name]['mean']]
g_means = [data['base']['g'][name]['mean'], data['r1']['g'][name]['mean'], data['r3']['g'][name]['mean'], data['r5']['g'][name]['mean']]
b_stds = [data['base']['b'][name]['std'], data['r1']['b'][name]['std'], data['r3']['b'][name]['std'], data['r5']['b'][name]['std']]
g_stds = [data['base']['g'][name]['std'], data['r1']['g'][name]['std'], data['r3']['g'][name]['std'], data['r5']['g'][name]['std']]

# 绘图
plt.figure(figsize=(6, 6))

# 绘制 test b 和 test g 的均值折线图
plt.plot(x_labels, b_means, label='Bias RCT', color='#ff0000', marker='o')
plt.plot(x_labels, g_means, label='Ground truth', color='#1673F2', marker='o')

# 绘制 test b 和 test g 的方差带
# plt.errorbar(x_labels, b_means, yerr=b_stds, label='Bias RCT', fmt='-o', color='#ff0000', capsize=5, elinewidth=2, markeredgewidth=2)
# plt.errorbar(x_labels, g_means, yerr=g_stds, label='Ground truth', fmt='-o', color='#1673F2', capsize=5, elinewidth=2, markeredgewidth=2)
plt.fill_between(x_labels, np.array(b_means) - np.array(b_stds), np.array(b_means) + np.array(b_stds), color='#ff0000', alpha=0.2)
plt.fill_between(x_labels, np.array(g_means) - np.array(g_stds), np.array(g_means) + np.array(g_stds), color='#1673F2', alpha=0.2)

base_mean_b = data['base']['b'][name]['mean']  # base test b 的均值
plt.axhline(y=base_mean_b, color='#D62728', linestyle='--', label='Bias RCT baseline')  # 红色虚线

base_mean_b = data['base']['g'][name]['mean']  # base test b 的均值
plt.axhline(y=base_mean_b, color='#1673F2', linestyle='--', label='Ground truth baseline')  # 红色虚线

# 设置图形的标题和标签
plt.title(f'n={number}')
plt.xlabel('Setting')
plt.ylabel(f'{name}')

# 显示图例
plt.legend()

# 显示图形
# plt.grid(True)
plt.tight_layout()
plt.show()
