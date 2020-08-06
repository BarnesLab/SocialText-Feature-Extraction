#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:34:08 2019

@author: sanjanamendu
"""

import pandas as pd

# =============================================================================
#                    LIWC (External Software Required)
# =============================================================================

df = pd.read_csv("liwc_fbmsg (participant+direction).csv") \
    .rename(columns={'Source (A)':'PID', 'Source (B)': 'Direction', 'Source (C)': 'Content'}) \
    .drop(['Content','WC'],1)

lexical_feat = pd.merge(df.loc[df.Direction == 'Incoming'].drop('Direction',1), 
                        df.loc[df.Direction == 'Outgoing'].drop('Direction',1), 
                        'left', on='PID', suffixes=["_incoming","_outgoing"]).reset_index()

lexical_feat.columns = lexical_feat.columns.str.lower()

lexical_feat.to_csv("lexical_feat.csv", index=False)