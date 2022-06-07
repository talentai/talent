import pandas as pd
import numpy as np

# Helper Functions Starts here #

def get_accuracy_chart(acc,acc_baseline):
    acc_format = round(acc*100,0)
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
                              acc_baseline,
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
                          "value":acc_format
                        }
                      ]
                    }
                  ]
                }
    return options