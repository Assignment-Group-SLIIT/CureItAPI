import pandas as pd
from flask import jsonify
from openpyxl import load_workbook

def readActivityDataFrame():
    activities_df = pd.read_excel('E:/SLIIT/Y4S1(Y4S2)/Research Project/PP2/hasani/Physical_Ac_Rec/Physical_Ac_Rec/activities.xlsx')
    return activities_df

def readUsersCompletionDataFrame():
    completerates_df = pd.read_excel('E:/SLIIT/Y4S1(Y4S2)/Research Project/PP2/hasani/Physical_Ac_Rec/Physical_Ac_Rec/completerate.xlsx')
    return completerates_df

def updateUserCompletionRates(data):
    userId = int(data['userId'])
    activityId = int(data['activityId'])
    complete_score = int(data['complete_score'])
    satisfaction_score = float(data['satisfaction_score'])

    wb_append = load_workbook("E:/SLIT/Y4S1(Y4S2)/Research Project/PP2/hasani/Physical_Ac_Rec/Physical_Ac_Rec/completeratecheckappend.xlsx")

    sheet = wb_append.active
    row = (userId,activityId, complet_score,satisfaction_score)
    sheet.append(row)
    wb_append.save('E:/SLIIT/Y4S1(Y4S2)/Research Project/PP2/hasani/Physical_Ac_Rec/Physical_Ac_Rec/completeratecheckappend.xlsx')
