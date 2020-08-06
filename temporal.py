#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:21:24 2019

@author: sanjanamendu
"""

from tqdm import tqdm
from itertools import product
import datetime as dt
import pandas as pd
import numpy as np
import os

home = os.path.expanduser("~")

df = pd.read_csv("fbmsg_agg.csv")
df['pd_time'] = pd.to_datetime(df['Timestamp'], unit='s')
                                
# =============================================================================
#                             Hourly Proportion
# =============================================================================

hours = [dt.time(i).strftime('%-I%p').lower() for i in range(24)]
participants = df.PID.unique().tolist()

# --- Bidirectional
hourly_msg_counts = df.groupby(['PID',df.pd_time.dt.hour])['Timestamp'].count().reset_index() # Number of messages sent by each participants divided up by hour of day
hourly_msg_prop = hourly_msg_counts # Number -> Proportion!
hourly_msg_prop['Timestamp'] = hourly_msg_counts.groupby('PID')['Timestamp'].apply(lambda x: x/x.sum())
hourly_msg_prop_bd = pd.DataFrame(list(product(participants, list(range(0,24)))), columns=['PID', 'pd_time'])
hourly_msg_prop_bd = hourly_msg_prop_bd.merge(hourly_msg_prop,on=['PID','pd_time'],how='outer').fillna(0)
hourly_msg_prop_bd = hourly_msg_prop_bd.pivot(index='PID',columns='pd_time',values='Timestamp').set_axis([x + "_bidirectional" for x in hours], axis=1, inplace=False)
hourly_msg_prop_bd.to_csv("fbmsg_hourlyprop.csv",index=False)

# --- Incoming
hourly_msg_counts = df[df.Incoming].groupby(['PID',df.pd_time.dt.hour])['Timestamp'].count().reset_index() # Number of messages sent by each participants divided up by hour of day
hourly_msg_prop = hourly_msg_counts # Number -> Proportion!
hourly_msg_prop['Timestamp'] = hourly_msg_counts.groupby('PID')['Timestamp'].apply(lambda x: x/x.sum())
hourly_msg_prop_in = pd.DataFrame(list(product(participants, list(range(0,24)))), columns=['PID', 'pd_time'])
hourly_msg_prop_in = hourly_msg_prop_in.merge(hourly_msg_prop,on=['PID','pd_time'],how='outer').fillna(0)
hourly_msg_prop_in = hourly_msg_prop_in.pivot(index='PID',columns='pd_time',values='Timestamp').set_axis([x + "_incoming" for x in hours], axis=1, inplace=False)
hourly_msg_prop_in.to_csv("hourly_msg_prop_incoming.csv",index=False)

# --- Outgoing
hourly_msg_counts = df[df.Outgoing].groupby(['PID',df.pd_time.dt.hour])['Timestamp'].count().reset_index() # Number of messages sent by each participants divided up by hour of day
hourly_msg_prop = hourly_msg_counts # Number -> Proportion!
hourly_msg_prop['Timestamp'] = hourly_msg_counts.groupby('PID')['Timestamp'].apply(lambda x: x/x.sum())
hourly_msg_prop_out = pd.DataFrame(list(product(participants, list(range(0,24)))), columns=['PID', 'pd_time'])
hourly_msg_prop_out = hourly_msg_prop_out.merge(hourly_msg_prop,on=['PID','pd_time'],how='outer').fillna(0)
hourly_msg_prop_out = hourly_msg_prop_out.pivot(index='PID',columns='pd_time',values='Timestamp').set_axis([x + "_outgoing" for x in hours], axis=1, inplace=False)
hourly_msg_prop_out.to_csv("hourly_msg_prop_outgoing.csv",index=False)

hourly_msg_prop = pd.merge(pd.merge(hourly_msg_prop_bd, hourly_msg_prop_in, 'left', on='PID'), hourly_msg_prop_out, 'left', on='PID').fillna(0)

# =============================================================================
#                               Latency
# =============================================================================

latency_df = pd.DataFrame(columns=['PID','incoming_latency','outgoing_latency'])
for pid, pid_df in tqdm(df.groupby('PID')):
    pil = []
    pol = []
    for grp, grp_df in pid_df.groupby('Group'):
        grp_df = grp_df.sort_values(by=['pd_time'])
        grp_df['sequence'] = grp_df.Outgoing.ne(grp_df.Outgoing.shift()).cumsum()
        
        seq_df = pd.DataFrame({'sequence': grp_df.sequence.unique(),
                               'timestamp':  grp_df.groupby('sequence').pd_time.first(),
                               'direction': grp_df.groupby('sequence').Outgoing.first().map({True:'Outgoing',False:'Incoming'})})
        seq_df['minutes_elapsed'] = seq_df.timestamp.diff().dt.seconds.fillna(0) / 60

        latency = seq_df.groupby('direction').minutes_elapsed.mean()
        
        pil.append(latency.Incoming if 'Incoming' in latency else 0)
        pol.append(latency.Outgoing if 'Outgoing' in latency else 0)

    latency_df = latency_df.append({'PID': pid, 'incoming_latency': np.mean(pil), 'outgoing_latency': np.mean(pol)}, ignore_index=True)
    latency_df.to_csv("fbmsg_latency.csv", index=False)

temporal_feat = pd.merge(latency_df, hourly_msg_prop, 'left', on='PID')

temporal_feat.to_csv("final/temporal_feat.csv", index=False)
