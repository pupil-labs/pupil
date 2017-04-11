'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import csv


def read_key_value_file(csvfile):
    """Reads CSV file, parses content into dict

    Args:
        csvfile (FILE): Readable file

    Returns:
        DICT: Dictionary containing file content
    """
    kvstore = {}  # init key value store
    first_line = csvfile.readline()
    if 'key' not in first_line or 'value' not in first_line:
        csvfile.seek(0)  # Seek to start if first_line is not an header
    dialect = csv.Sniffer().sniff(first_line, delimiters=',\t')
    reader = csv.reader(csvfile, dialect)  # create reader
    for row in reader:
        kvstore[row[0]] = row[1]
    return kvstore


def write_key_value_file(csvfile,dictionary,append=False):
    """Writes a dictionary to a writable file in a CSV format

    Args:
        csvfile (FILE): Writable file
        dictionary (dict): Dictionary containing key-value pairs
        append (bool, optional): Writes `key,value` as fieldnames if False

    Returns:
        None: No return
    """
    writer = csv.writer(csvfile, delimiter=',')
    if not append:
        writer.writerow(['key','value'])
    for key,val in dictionary.items():
        writer.writerow([key,val])

if __name__ == '__main__':
    test = {'foo':'bar','oh':'rl","y','it was':'not me'}
    test_append = {'jo':'ho'}
    test_updated = test.copy()
    test_updated.update(test_append)

    testfile = '.test.csv'

    # Test write+read
    with open(testfile, 'w') as csvfile:
        write_key_value_file(csvfile,test)
    with open(testfile, 'r') as csvfile:
        result = read_key_value_file(csvfile)
    assert test == result

    # Test write+append (same keys)+read
    with open(testfile, 'w') as csvfile:
        write_key_value_file(csvfile,test)
        write_key_value_file(csvfile,test,append=True)
    with open(testfile, 'r') as csvfile:
        result = read_key_value_file(csvfile)
    assert test == result

    # Test write+append (different keys)+read
    with open(testfile, 'w') as csvfile:
        write_key_value_file(csvfile,test)
        write_key_value_file(csvfile,test_append,append=True)
    with open(testfile, 'r') as csvfile:
        result = read_key_value_file(csvfile)
    assert test_updated == result

    import os
    os.remove(testfile)
    print('CSV Test: successful')