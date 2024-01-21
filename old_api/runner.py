import pickle
from typing import Any, Union
import pandas as pd

# Dictionary to get data from street name
# street_data = {
#     road name : [ intersection, public places, ways, label]
# }
street_data: dict[Union[str, Any], Union[list[int], Any]] = {
    "Adenwala Rd": [16, 14, 2, 0],
    "Adenwala Rd and Nathalal Parekh Marg": [9, 13, 2, 1],
    "Antop Hill Rd and Shaikh Misree Marg": [4, 3, 7, 2],
    "Balaram Babu Khedekar Marg": [8, 5, 1, 3],
    "Bharni Naka Rd/Sir Pochkhan Wala Rd/Vidyalankar College Rd": [2, 12, 2, 4],
    "Comrade Harbanslal Marg/Flank Rd": [9, 19, 1, 5],
    "Comrade Harbanslal Marg/Flank Rd and Dr Baba Saheb Ambedkar Rd": [9, 19, 1, 6],
    "Dadar TT Flyover/Eastern Express Hwy": [5, 8, 2, 7],
    "David S Barretto Rd": [6, 4, 2, 8],
    "David S Barretto Rd and Barkat Ali Dargah Rd": [9, 12, 2, 9],
    "Dr Baba Saheb Ambedkar Rd": [15, 19, 1, 10],
    "Eastern Express Hwy and Dr Baba Saheb Ambedkar Rd": [4, 6, 2, 11],
    "GD Ambekar Marg/Katrak Rd": [2, 5, 2, 12],
    "GD Ambekar Marg/Katrak Rd and Firdausi Rd": [8, 8, 2, 13],
    "Govindji Keni Rd": [4, 13, 1, 14],
    "Govindji Keni Rd and GD Ambekar Marg/Katrak Rd": [11, 5, 1, 15],
    "Govindji Keni Rd, Mahatma Jyotiba Phule Rd/Naigaon Rd and BJ Deorukhkar Marg/BJ Devrukhkar Rd": [1, 6, 1, 16],
    "H R Mahajani Rd": [1, 1, 2, 17],
    "Jerbai Wadia Rd": [5, 11, 1, 18],
    "Kings Cir/Matunga Cir and Puranmal Singhani Rd": [7, 19, 1, 19],
    "Lady Jamshedji Rd/LJ Rd and N C. Kelkar Rd": [8, 9, 2, 20],
    "Lady Jehangir Rd": [5, 6, 1, 21],
    "Mahatma Jyotiba Phule Rd/Naigaon Rd and MMGS Marg/Naigaon Cross Rd": [3, 32, 2, 22],
    "Mancherji Joshi Rd": [3, 4, 2, 23],
    "N C. Kelkar Rd and Lady Jamshedji Rd": [13, 38, 2, 24],
    "Nathalal Parekh Marg and Rafi Ahmed Kidwai Rd": [11, 14, 2, 25],
    "Puranmal Singhani Rd and Nathalal Parekh Marg": [2, 6, 2, 26],
    "Rafi Ahmed Kidwai Rd": [10, 8, 2, 27],
    "Rd Number 16 and Barkat Ali Dargah Rd": [6, 7, 2, 28],
    "SM Uphill Rd": [2, 8, 2, 29],
    "Shaikh Misree Marg": [10, 30, 1, 30],
    "Shaikh Misree Marg and JK Bhasin Marg": [11, 35, 1, 31],
    "Shaikh Misree Marg and SM Uphill Rd": [6, 12, 1, 32],
    "Taikalwadi Marg and JK Sawant Marg": [8, 9, 2, 33],
    "Tilak Rd": [4, 16, 2, 34],
    "Tilak Rd and Firdausi Rd": [3, 4, 2, 35],
    "Tilak Rd/Tilak Bridge/Tilak Flyover": [1, 13, 2, 36],
}


def run():
    data = pd.read_csv('calculationdata.csv')
    data = data.astype(str)

    # street = "{}".format(parameters.get('street'))
    street = "Adenwala Rd"
    # data_dict = street_data[street]
    
    # time = "{}".format(parameters.get('time'))
    # hour = int(time_opt(time))

    hour = 12

    # date = "{}".format(parameters.get('date'))
    data_dict = [16, 14, 2, 0]

    label_enc_road = "{}".format(data_dict[3])

    # day = date_to_day(date)
    day = 3  # 0 to 6 or 1 to 7 as Mon to Sun

    model = pickle.load(open('catboost_version2.pkl', 'rb'))

    model_result = model.predict([data_dict[0], data_dict[1], data_dict[2], data_dict[3], hour, day])
    print(model_result)
    result = "{}".format(model_result[0])
    print(result)

run()
