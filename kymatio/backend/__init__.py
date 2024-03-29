# Backend decision logic at import
# We will have backend fixed at import until we come up with something more
# elegant.

# Here's how it is decided upon:

# 1. Is there an environment variable SCATTERING_BACKEND_2D?
# 2. Is there an environment variable SCATTERING_BACKEND?
# 3. Is there a config file? If so, go and find the backend entry
# 4. Set the backend to DEFAULT_BACKEND

DEFAULT_BACKEND = "torch"
#DEFAULT_BACKEND = "skcuda"

import os
import configparser
import appdirs


# find config file

config_file = os.path.join(appdirs.user_config_dir("scattering"), "scattering.cfg")
cp = configparser.ConfigParser()

if os.path.exists(config_file):
    cp.read(config_file)
    BACKEND = cp.get('general', 'backend_2d', 
                     fallback=cp.get('general', 'backend', fallback=None))
    if BACKEND is None:
        BACKEND = DEFAULT_BACKEND
        if 'general' not in cp.sections():
            cp.add_section('general')
        cp['general']['backend_2d'] = BACKEND
        try:
            with open(config_file, "w") as f:
                cp.write(f)
        except:
            pass

else:
    BACKEND = DEFAULT_BACKEND
    # try to write config file
    try:
        dirname = os.path.dirname(config_file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        cp.add_section('general')
        cp['general']['backend_2d'] = BACKEND
        with open(config_file, "w") as f:
            cp.write(f)
    except:
        pass



# general env:
BACKEND = os.environ.get("KYMATIO_BACKEND", BACKEND)
#BACKEND = 'skcuda'

# 2d specific env:
#BACKEND = os.environ.get("SCATTERING_BACKEND_2D", BACKEND)

from .backend_common import *

if BACKEND == 'torch':
    print('use torch backend')
    from .backend_torch import *
elif BACKEND == 'skcuda':
    print('use skcuda backend')
    from .backend_skcuda import *
else:
    # For now, raise and error
    raise ValueError("Backend {} unknown".format(BACKEND))


