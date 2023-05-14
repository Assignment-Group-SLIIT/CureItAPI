import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.data.dataset import readActivityDataFrame, readUsersCompletionDataFrame


def collabUser(propId):
    activities_df = readActivityDataFrame()
    completerates_df = readUsersCompletionDataFrame()

    #obtaining the mean value based on satisfaction score and the completion score to compute the preference avg.
    #Then merge it to a matrix of completion rates
    Mean = completerates_df.groupby(by='userId',as_index=False)['satisfaction_score','complete_score'].mean()
    Rating_avg = pd.merge(completerates_df,Mean,on='userId')
    Rating_avg['adg_rating']=Rating_avg['satisfaction_score_x']-Rating_avg['satisfaction_score_y'] + Rating_avg['complete_score_x']-Rating_avg['complete_score_y']

    #ploting table with user Id vs activity Id with matrix values representing satisfaction_score
    check = pd.pivot_table(Rating_avg,values='satisfaction_score_x',index='userId',columns='activityId')

    #ploting table with user Id vs activity Id with matrix values representing completion_score
    check2 = pd.pivot_table(Rating_avg,values='complete_score_x',index='userId',columns='activityId')

    #ploting table with user Id vs activity Id with matrix values representing mean preference
    final = pd.pivot_table(Rating_avg,values='adg_rating',index='userId',columns='activityId')

    # Replacing NaN by Activity Average
    final_activity = final.fillna(final.mean(axis=0))

    # Replacing NaN by user Average
    final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)

    # user similarity on replacing NAN by user avg
    b = cosine_similarity(final_user)
    np.fill_diagonal(b, 0 )
    similarity_with_user = pd.DataFrame(b,index=final_user.index)
    similarity_with_user.columns=final_user.index

    # user similarity on replacing NAN by item(activity) avg
    cosine = cosine_similarity(final_activity)
    np.fill_diagonal(cosine, 0 )
    similarity_with_activity = pd.DataFrame(cosine,index=final_activity.index)
    similarity_with_activity.columns=final_user.index

    #finding the nearest neighbor
    def find_n_neighbours(df,n):
        order = np.argsort(df.values, axis=1)[:, :n]
        df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
            .iloc[:n].index, 
            index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
        return df

    # top 20 neighbours for each user based on user similarity
    sim_user_5_u = find_n_neighbours(similarity_with_user,20)

    # top 20 neighbours for based on activity preference similarity
    sim_user_5_m = find_n_neighbours(similarity_with_activity,20)

    #returning the activities of the similar users
    def get_user_similar_activities( user1, user2 ):
        common_activities = Rating_avg[Rating_avg.userId == user1].merge(
        Rating_avg[Rating_avg.userId == user2],
        on = "activityId",
        how = "inner" )
        return common_activities.merge( activities_df, on = 'activityId' )


    a = get_user_similar_activities(1000,1017)
    a = a.loc[ : , ['satisfaction_score_x_x','satisfaction_score_x_y','complete_score_x_x','complete_score_x_y','title']]

    def User_item_score(user,item):
        a = sim_user_5_m[sim_user_5_m.index==user].values
        b = a.squeeze().tolist()
        c = final_activity.loc[:,item]
        d = c[c.index.isin(b)]
        f = d[d.notnull()]
        avg_user = Mean.loc[Mean['userId'] == user,'satisfaction_score'].values[0]
        avg_user = Mean.loc[Mean['userId'] == user,'complete_score'].values[0]
        index = f.index.values.squeeze().tolist()
        corr = similarity_with_activity.loc[user,index]
        fin = pd.concat([f, corr], axis=1)
        fin.columns = ['adg_score','correlation']
        fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)
        nume = fin['score'].sum()
        deno = fin['correlation'].sum()
        final_score = avg_user + (nume/deno)
        return final_score

    score = User_item_score(20,35)
    print("score (u,i) is",score)

    Rating_avg = Rating_avg.astype({"activityId": str})
    Activity_user = Rating_avg.groupby(by = 'userId')['activityId'].apply(lambda x:','.join(x))

    #user score calculations
    def User_item_score1(user):
        Activity_attempted_by_user = check.columns[check[check.index==user].notna().any()].tolist()
        a = sim_user_5_m[sim_user_5_m.index==user].values
        b = a.squeeze().tolist()
        d = Activity_user[Activity_user.index.isin(b)]
        l = ','.join(d.values)
        Activity_attempted_by_similar_users = l.split(',')
        
        N = 5
        Most_recent_actvities_attempted_by_user = Activity_attempted_by_user[-N:]
        
        #Activities under consideration but disregarding the last 5 attempted activities
        Activities_under_consideration = list(set(Activity_attempted_by_similar_users)-set(list(map(str, Most_recent_actvities_attempted_by_user))))
        Activities_under_consideration = list(map(int, Activities_under_consideration)) 
                                                                                        
        # print(Activity_attempted_by_user)
        # print(Most_recent_actvities_attempted_by_user)
        # print(Activities_under_consideration)

        score = []
        for item in Activities_under_consideration:
            c = final_activity.loc[:,item]
            d = c[c.index.isin(b)]
            f = d[d.notnull()]
            avg_user = Mean.loc[Mean['userId'] == user,'satisfaction_score'].values[0]
            avg_user = Mean.loc[Mean['userId'] == user,'complete_score'].values[0]
            index = f.index.values.squeeze().tolist()
            corr = similarity_with_activity.loc[user,index]
            fin = pd.concat([f, corr], axis=1)
            fin.columns = ['adg_score','correlation']
            fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)
            nume = fin['score'].sum()
            deno = fin['correlation'].sum()
            final_score = avg_user + (nume/deno)
            score.append(final_score)
        data = pd.DataFrame({'activityId':Activities_under_consideration,'score':score})
        top_5_recommendation = data.sort_values(by='score',ascending=False).head(5)
        Activity_Name = top_5_recommendation.merge(activities_df, how='inner', on='activityId')
    #     Activity_Names = Activity_Name.title.values.tolist()
        Activity_Names = Activity_Name.activityId.tolist()
        return Activity_Names
    
    return User_item_score1(propId)

    # user = int(input("Enter the user id to whom you want to recommend : "))
    # predicted_activities = User_item_score1(user)
    # print(" ")
    # print("The Recommendations for User Id : ", user)
    # print("   ")
    # for i in predicted_activities:
    #     print(i)
    # activity_considered = i
    # print(activity_considered)