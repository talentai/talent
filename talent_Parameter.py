import pandas as pd
import numpy as np

# Helper Functions Starts here #


# Set Hydralit Components - can apply customisation to almost all the properties of the card, including the progress bar
def get_hc_theme(style):
    
    output_theme = np.nan
    if style == 'info':
        output_theme = {'bgcolor': '#EFF8F7','title_color': 'green','content_color': 'green','icon_color': 'green', 'icon': 'fa fa-check-circle'}
    elif style == 'female':
        output_theme = {'bgcolor': '#f8f8fa','title_color': 'black','content_color': 'black','icon_color': '#0893cf', 'icon': 'fa fa-female'}
    elif style == 'warning':
        output_theme = {'bgcolor': '#f9f9f9','title_color': 'orange','content_color': 'black','icon_color': 'orange', 'icon': 'fa fa-exclamation-circle'}
    elif style == 'good':
        # output_theme = {'font-size': '0.5rem','bgcolor': '#EFF8F7','title_color': 'green','content_color': 'black','icon_color': 'green', 'icon': 'fa fa-check-circle'}
        output_theme = {'font-size': '0.5rem','bgcolor': '#EFF8F7','title_color': 'green','content_color': 'black','icon_color': 'green'}
    
    else:
        output_theme = np.nan
    return output_theme


def get_r2_option(r2):
    r2_format = round(r2*100,0)
    options = {
                  "series":[
                    {"max":100,
                     "min":0,
                     "splitNumber": 10,
                      "type":"gauge",
                      "axisLine":{
                        "lineStyle":{
                          "width":10,
                          "color":[
                            [
                              0.7,
                              "#9ca5af"
                            ],
                            [
                              1,
                              "#6bd47c"
                            ]
                          ]
                        }
                      },
                      "pointer":{
                        "itemStyle":{
                          "color":"auto"
                        }
                      },
                      "axisTick":{
                        "distance":-30,
                        "length":2,
                        "lineStyle":{
                          "color":"#fff",
                          "width":2
                        }
                      },
                      "splitLine":{
                        "distance":-5,
                        "length":10,
                        "lineStyle":{
                          "color":"#fff",
                          "width":1
                        }
                      },
                      "axisLabel":{
                        "color":"auto",
                        "distance":10,
                        "fontSize":10
                      },
                      "detail":{
                        "valueAnimation":True,
                        "formatter":"{value}%",
                        "color":"auto",
                        "fontSize": 15
                      },
                      "data":[
                        {
                          "value":r2_format
                        }
                      ]
                    }
                  ]
                }
    return options

def get_gender_gap_option(gap):
    gap_format = round(gap*100,0)
    options = {
                  "series":[
                    {"max":20,
                     "min":-20,
                     "splitNumber": 8,
                      "type":"gauge",
                      "axisLine":{
                        "lineStyle":{
                          "width":10,
                          "color":[
                              [
                              0.375,
                              "#9ca5af"
                            ],
                            [
                              1,
                              "#6bd47c"
                            ]
                          ]
                        }
                      },
                      "pointer":{
                        "itemStyle":{
                          "color":"auto"
                        }
                      },
                      "axisTick":{
                        "distance":-30,
                        "length":2,
                        "lineStyle":{
                          "color":"#fff",
                          "width":2
                        }
                      },
                      "splitLine":{
                        "distance":-5,
                        "length":10,
                        "lineStyle":{
                          "color":"#fff",
                          "width":1
                        }
                      },
                      "axisLabel":{
                        "color":"auto",
                        "distance":10,
                        "fontSize":10
                      },
                      "detail":{
                        "valueAnimation":True,
                        "formatter":"{value}%",
                        "color":"auto",
                        "fontSize": 15
                      },
                      "data":[
                        {
                          "value":gap_format
                        }
                      ]
                    }
                  ]
                }
    return options

