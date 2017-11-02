# Assignment 3, Part 3: Pull data about alternative fuel stations

from urllib2 import urlopen, Request
import json
import pandas as pd


def get_data(url):
    """ Get response in JSON from URL and convert it into Python dict. """

    # use urlopen() to get a response from the server
    response = urlopen(Request(url))

    # get JSON text from server response
    raw_data = response.read().decode('utf-8')

    # convert JSON text into Python object
    data = json.loads(raw_data)

    return data


# TASK 3.1
# get your private API key via the web form and paste the 40 characters below
api_key = 'e4Q7TieJUy2zTaB6ubxekqa40SBO8adqzbPczvDr'

# set location as Ann Arbor
# note that %20 in URLs means the space character
location = 'Ann%20Arbor+MI'

# set up the query format
query_fmt = 'https://api.data.gov/nrel/alt-fuel-stations/v1/nearest.json?api_key=%s&location=%s'

# TASK 3.2
# construct the query URL using api_key, location, query_fmt
url = query_fmt.replace('%s', api_key, 1).replace('%s', location, 1)

# get data from the given URL
print 'Fetching results from URL: %s' % url
data = get_data(url)

# TASK 3.3
# get the number of results from one of the keys in the Python dict data
nresults = data['total_results']
print 'There are a total of %d results' % nresults

# TASK 3.4
# another key in the dict will be a list of fuel stations
fs_list = data['fuel_stations']
print 'Have data on %d results' % len(fs_list)

# TASK 3.5
# Fetching additional fuel station data
while len(fs_list) < nresults:
    # TASK 3.5.1
    # construct a new query URL now with an offset parameter
    offset = 1 #nresults - len(fs_list)
    url_offset = url + '&offset=' + str(offset)
    
    # fetch additional results using the specified offset
    print 'Fetching additional results from URL: %s' % url_offset
    data = get_data(url_offset)
    
    # TASK 3.5.2
    # extend the list with the newly fetched fuel station data
    fs_list.extend(data['fuel_stations'])
    print 'Now have data on %d results' % len(fs_list)

# TASK 3.6
df = pd.DataFrame(fs_list)
n, p = df.shape
print "Created data frame with data on %d variables about %d fuel stations\n" % (p, n)

# print a few columns of the DataFrame
print df.to_string(columns=['station_name', 'street_address', 'fuel_type_code'])
