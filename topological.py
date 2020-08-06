#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:11:48 2019

@author: sanjanamendu
"""

import pandas as pd
import numpy as np
import os

home = os.path.expanduser("~")

df = pd.read_csv("fbmsg_agg.csv")

df['Conversation Type'] = (df['Group Size'] == 1).map({True: 'Individual', False: 'Group'}) # Conversation Type (i.e. Individual or Group)
grp_sizes = df.groupby(['PID','Group','Group Size']).first().reset_index()[['PID','Group','Group Size']] # Group Size

# =============================================================================
#                             Edge Weights  
# =============================================================================

# --- Bidirectional
freq = df.groupby('PID').Group.value_counts().rename('Frequency').reset_index()
freq['Prop'] = freq.groupby('PID').Frequency.apply(lambda x: x/x.sum())
freq['Entropy'] = freq.groupby('PID').Prop.apply(lambda x: -(x)*np.log(x))
grp_var_bd = freq.groupby('PID').Prop.max().rename('max_bidirectional').reset_index().set_index('PID')
grp_var_bd['Max Group'] = freq[freq.groupby('PID').Prop.transform(max) == freq.Prop].drop_duplicates(subset='PID', keep='first').set_index('PID').Group
grp_var_bd = grp_var_bd.reset_index() \
    .merge(grp_sizes, how="left", left_on=["PID","Max Group"], right_on=["PID","Group"]) \
    .drop(['Max Group','Group'],1) \
    .rename(columns={'Group Size':'max_group_size_bidirectional'}).set_index('PID')
grp_var_bd['mean_chat_entropy_bidirectional'] = freq.groupby('PID').Entropy.mean()
grp_var_bd.to_csv("fbmsg_grp_prop (incoming).csv")

# --- Incoming
freq = df[df.Incoming].groupby('PID').Group.value_counts().rename('Frequency').reset_index()
freq['Prop'] = freq.groupby('PID').Frequency.apply(lambda x: x/x.sum())
freq['Entropy'] = freq.groupby('PID').Prop.apply(lambda x: -(x)*np.log(x))
grp_var_in = freq.groupby('PID').Prop.max().rename('max_incoming').reset_index().set_index('PID')
grp_var_in['Max Group'] = freq[freq.groupby('PID').Prop.transform(max) == freq.Prop].drop_duplicates(subset='PID', keep='first').set_index('PID').Group
grp_var_in = grp_var_in.reset_index() \
    .merge(grp_sizes, how="left", left_on=["PID","Max Group"], right_on=["PID","Group"]) \
    .drop(['Max Group','Group'],1) \
    .rename(columns={'Group Size':'max_group_size_incoming'}).set_index('PID')
grp_var_in['mean_chat_entropy_incoming'] = freq.groupby('PID').Entropy.mean()
grp_var_in.to_csv("fbmsg_grp_prop (incoming).csv")

# --- Outgoing
freq = df[df.Outgoing].groupby('PID').Group.value_counts().rename('Frequency').reset_index()
freq['Prop'] = freq.groupby('PID').Frequency.apply(lambda x: x/x.sum())
freq['Entropy'] = freq.groupby('PID').Prop.apply(lambda x: -(x)*np.log(x))
grp_var_out = freq.groupby('PID').Prop.max().rename('max_outgoing').reset_index().set_index('PID')
grp_var_out['Max Group'] = freq[freq.groupby('PID').Prop.transform(max) == freq.Prop].drop_duplicates(subset='PID', keep='first').set_index('PID').Group
grp_var_out = grp_var_out.reset_index() \
    .merge(grp_sizes, how="left", left_on=["PID","Max Group"], right_on=["PID","Group"]) \
    .drop(['Max Group','Group'],1) \
    .rename(columns={'Group Size':'max_group_size_outgoing'}).set_index('PID')
grp_var_out['mean_chat_entropy_outgoing'] = freq.groupby('PID').Entropy.mean()
grp_var_out.to_csv("fbmsg_grp_prop (incoming).csv")

grp_var = pd.merge(pd.merge(grp_var_bd, grp_var_in, 'left', on='PID'), grp_var_out, 'left', on='PID').fillna(0).reset_index()

# =============================================================================
#                    Number of Alters (Individual + Group)
# =============================================================================

# --- Bidirectional
group_type_counts = df.groupby(['PID','Conversation Type'])['Group'].nunique().reset_index()
fbmsg_grps_bd = pd.DataFrame({'PID': group_type_counts.PID.unique()})
fbmsg_grps_bd = fbmsg_grps_bd.merge(group_type_counts[group_type_counts['Conversation Type'] == 'Individual'],'left',on='PID').drop('Conversation Type',1).rename(columns={'Group':'individual_bidirectional'})
fbmsg_grps_bd = fbmsg_grps_bd.merge(group_type_counts[group_type_counts['Conversation Type'] == 'Group'],'left',on='PID').drop('Conversation Type',1).rename(columns={'Group':'group_bidirectional'})
fbmsg_grps_bd.to_csv("fbmsg_grps.csv",index=False)

# --- Incoming
group_type_counts = df[df.Incoming].groupby(['PID','Conversation Type'])['Group'].nunique().reset_index()
fbmsg_grps_in = pd.DataFrame({'PID': group_type_counts.PID.unique()})
fbmsg_grps_in = fbmsg_grps_in.merge(group_type_counts[group_type_counts['Conversation Type'] == 'Individual'],'left',on='PID').drop('Conversation Type',1).rename(columns={'Group':'individual_incoming'})
fbmsg_grps_in = fbmsg_grps_in.merge(group_type_counts[group_type_counts['Conversation Type'] == 'Group'],'left',on='PID').drop('Conversation Type',1).rename(columns={'Group':'group_incoming'})
fbmsg_grps_in.to_csv("fbmsg_grps_incoming.csv",index=False)

# --- Outgoing
group_type_counts = df[df.Outgoing].groupby(['PID','Conversation Type'])['Group'].nunique().reset_index()
fbmsg_grps_out = pd.DataFrame({'PID': group_type_counts.PID.unique()})
fbmsg_grps_out = fbmsg_grps_out.merge(group_type_counts[group_type_counts['Conversation Type'] == 'Individual'],'left',on='PID').drop('Conversation Type',1).rename(columns={'Group':'individual_outgoing'})
fbmsg_grps_out = fbmsg_grps_out.merge(group_type_counts[group_type_counts['Conversation Type'] == 'Group'],'left',on='PID').drop('Conversation Type',1).rename(columns={'Group':'group_outgoing'})
fbmsg_grps_out.to_csv("fbmsg_grps_outgoing.csv",index=False)

fbmsg_grps = pd.merge(pd.merge(fbmsg_grps_bd, fbmsg_grps_in, 'left', on='PID'), fbmsg_grps_out, 'left', on='PID').fillna(0)

topological_feat = pd.merge(grp_var, fbmsg_grps, 'left', on='PID')
topological_feat.to_csv("final/topological_feat.csv", index=False)
