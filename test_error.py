import sys, traceback
sys.path.append('C:\\Users\\trund\\QRT\\Strategies')
from get_nasdaq_tickers import get_nasdaq_tickers
try:
    get_nasdaq_tickers()
except Exception as e:
    print('====== EXCEPTION START ======')
    traceback.print_exc()
    print('====== EXCEPTION END ======')
