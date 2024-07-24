
from __future__ import absolute_import
from __future__ import print_function


import os
from subprocess import call
import sys
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME'")

netgenBinary = checkBinary('netgenerate')
jtrrouterBinary = checkBinary('jtrrouter')
sumoBinary = checkBinary('sumo')
import randomTrips  # noqa

# call([netgenBinary, '-c', 'data/manhattan.netgcfg'])
randomTrips.main(randomTrips.get_options([
    '--flows', '500',
    '-b', '0',
    '-e', '1800',
    '-n', 'net.net.xml',
    '-o', 'flows.xml',
    '--jtrrouter',
    '--trip-attributes', 'departPos="random" departSpeed="max"']))
call([jtrrouterBinary, '-c', 'berlin.jtrrcfg'])

call([sumoBinary, '-c', 'berlin.sumocfg', '--duration-log.statistics', '-e' '100'])
