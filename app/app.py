import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import yfinance as yf
import plotly.graph_objects as go
import time

st.set_page_config(page_title='Wealth Survival Score', page_icon='📊', layout='centered')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_rf():
    with open(os.path.join(BASE_DIR,'..','models','rf_model.pkl'),'rb') as f:
        return pickle.load(f)

@st.cache_data(ttl=3600)
def load_signals():
    try:
        tickers={'Gold':'GC=F','Oil':'CL=F','Wheat':'ZW=F','Copper':'HG=F','VIX':'^VIX'}
        data={}
        for name,ticker in tickers.items():
            df=yf.download(ticker,period='6mo',interval='1mo',progress=False,auto_adjust=True)
            if isinstance(df.columns,pd.MultiIndex): df.columns=[c[0] for c in df.columns]
            if not df.empty: data[name]=float(df['Close'].iloc[-1])
        try:
            from pytrends.request import TrendReq
            pt=TrendReq(hl='en-US',tz=360)
            pt.build_payload(['stock market crash'],timeframe='today 3-m',geo='',gprop='')
            td=pt.interest_over_time()
            fear=float(td['stock market crash'].iloc[-1]) if not td.empty else 35.0
        except:
            fear=35.0
        master=pd.read_csv(os.path.join(BASE_DIR,'..','data','raw','master_signals.csv'),index_col=0,parse_dates=True)
        def norm(v,c):
            mn,mx=master[c].min(),master[c].max()
            return float(np.clip((v-mn)/(mx-mn)*100,0,100))
        return {'Gold':norm(data.get('Gold',1900),'Gold'),'Oil':norm(data.get('Oil',80),'Oil'),
                'Wheat':norm(data.get('Wheat',600),'Wheat'),'Copper':norm(data.get('Copper',4.0),'Copper'),
                'VIX_score':norm(data.get('VIX',20),'VIX_score'),'fear_composite':float(np.clip(fear,0,100))}
    except:
        return {'Gold':72.0,'Oil':58.0,'Wheat':45.0,'Copper':61.0,'VIX_score':55.0,'fear_composite':48.0}

def compute(country,age,s,co,ca,re,beh,sig,rf):
    feats=['VIX_score','Gold','Oil','fear_composite','Wheat','Copper']
    X=np.array([[sig[f] for f in feats]])
    pr=rf.predict_proba(X); cl=list(rf.classes_)
    sp=pr[0][cl.index(1)] if 1 in cl else 0.3
    cr={'Germany':0.72,'United States':0.58,'United Kingdom':0.63,'India':0.45,'France':0.67,'Japan':0.70,'Brazil':0.38}
    cm=cr.get(country,0.60)
    dp={'Germany':1.0,'United States':0.468,'United Kingdom':0.539,'India':0.0,'France':0.632,'Japan':1.0,'Brazil':0.15}
    demo=dp.get(country,0.5)
    ms=(1-sp)*100; cs=cm*100; ds=(1-demo)*100
    mx=max(s,co,ca,re)/100
    if mx>0.90: als=10
    elif mx>0.80: als=28
    elif ca<5: als=32
    elif ca>=20 and co>=15: als=95
    elif ca>=15 and co>=10: als=80
    elif ca>=10 and co>=5: als=65
    elif ca>=10: als=52
    else: als=42
    bs={'I panic-sold everything':10,'I sold some, held some':35,'I held and watched in pain':55,'I bought the dip':90,'I was not invested yet':48}
    bsc=bs.get(beh,50)
    ap={'18 - 25':0,'26 - 35':0,'36 - 45':-3,'46 - 55':-8,'56 - 65':-15,'65+':-22}
    apen=ap.get(age,0)
    raw=ms*0.25+cs*0.20+ds*0.15+als*0.30+bsc*0.10
    score=int(np.clip(raw+apen,5,98))
    if score>=75: lbl,col='Strong resilience','#1D9E75'
    elif score>=55: lbl,col='Moderate resilience','#EF9F27'
    elif score>=35: lbl,col='Fragile','#E24B4A'
    else: lbl,col='High vulnerability','#C0392B'
    ss={'Central bank language drift':int(min(sig['VIX_score']*0.6+sp*40,100)),
        'Commodity stress index':int(min(sig['Gold']*0.5+sig['Oil']*0.5,100)),
        'Geopolitical contagion':int(min(sig['fear_composite']*0.7+(1-cm)*30,100)),
        'Demographic pressure':int(demo*100),
        'Retail capitulation risk':int(round(sp*100))}
    contrarian_desc = 'Highest survival rate - and your {:.0f}% cash gives you the firepower to execute.'.format(ca) if ca>=15 else 'Highest survival rate - but your {:.0f}% cash may not be enough when the dip arrives.'.format(ca)
    anxious_desc = 'You hold through crashes - your {:.0f}% cash buffer helps but pre-commitment rules will stop emotional exits.'.format(ca) if ca>=15 else 'Emotional exits cost you 12% vs systematic rebalancers - and your low cash leaves you exposed.'
    am={'I panic-sold everything':('The Panic Seller','You exit at the worst moment. Build a cash buffer of at least 20% as your psychological anchor.'),
        'I sold some, held some':('The Reluctant Holder','You partially protect yourself but leave recovery gains on the table. A written rebalancing rule fixes this.'),
        'I held and watched in pain':('The Anxious Holder', anxious_desc),
        'I bought the dip':('The Contrarian', contrarian_desc),
        'I was not invested yet':('The Observer','No crash trauma - but entry timing is your critical risk. Define your entry rules now.')}
    al,ad=am.get(beh,('The Holder','Steady under pressure.'))
    return {'score':score,'label':lbl,'color':col,'stress':int(round(sp*100)),'signals':ss,'archetype':al,'arch_desc':ad}

def bar(label,val):
    c='#E24B4A' if val>70 else '#EF9F27' if val>50 else '#1D9E75'
    s='High risk' if val>70 else 'Elevated' if val>50 else 'Stable'
    st.markdown(f'<div style="margin-bottom:8px"><div style="display:flex;justify-content:space-between;font-size:13px"><span>{label}</span><span style="color:{c};font-weight:500">{s} ({val})</span></div><div style="background:#f0f0f0;border-radius:4px;height:7px"><div style="width:{val}%;background:{c};height:7px;border-radius:4px"></div></div></div>',unsafe_allow_html=True)

st.markdown('## Will your financial life survive the next 10 years?')
st.markdown('*30 seconds. No sign-up. Brutally honest.*')
st.divider()

c1,c2=st.columns(2)
country=c1.selectbox('Your country',['Germany','United States','United Kingdom','India','France','Japan','Brazil'])
age=c2.selectbox('Your age bracket',['18 - 25','26 - 35','36 - 45','46 - 55','56 - 65','65+'],index=2)
beh=st.selectbox('How did you feel during the 2022 market crash?',['I panic-sold everything','I sold some, held some','I held and watched in pain','I bought the dip','I was not invested yet'])
st.selectbox('Biggest fear for the next decade?',['Inflation destroying my savings','A major stock market crash','Geopolitical war disrupting markets','AI making my skills worthless','Climate collapse hitting commodities'])

st.markdown('**Portfolio allocation — enter percentages below. Real estate is auto-calculated to keep total at exactly 100%.**')

n1,n2,n3=st.columns(3)
s_val=n1.number_input('Stocks / ETFs %',min_value=0,max_value=100,value=60,step=5)
co_val=n2.number_input('Commodities %',min_value=0,max_value=100,value=15,step=5)
ca_val=n3.number_input('Cash / Bonds %',min_value=0,max_value=100,value=15,step=5)
re_val=100-s_val-co_val-ca_val

ca2,cb2,cc2,cd2=st.columns(4)
ca2.metric('Stocks',f'{s_val}%')
cb2.metric('Commodities',f'{co_val}%')
cc2.metric('Cash',f'{ca_val}%')
cd2.metric('Real estate (auto)',f'{re_val}%')

if re_val<0:
    st.error(f'Total is {s_val+co_val+ca_val}%. Reduce your inputs by {abs(re_val)}% to continue.')
    valid=False
else:
    st.success(f'Total: 100% locked  |  Stocks {s_val}%  Commodities {co_val}%  Cash {ca_val}%  Real estate {re_val}%')
    valid=True

st.divider()
if st.button('Calculate my Wealth Survival Score',use_container_width=True,type='primary',disabled=not valid):
    with st.spinner('Computing...'):
        prog=st.progress(0,text='Fetching live data...')
        time.sleep(0.3); prog.progress(30,text='Loading signals...')
        sig=load_signals()
        prog.progress(65,text='Running model...')
        rf=load_rf()
        prog.progress(90,text='Calculating score...')
        res=compute(country,age,s_val,co_val,ca_val,re_val,beh,sig,rf)
        prog.progress(100); time.sleep(0.2); prog.empty()

    st.markdown(f'<div style="text-align:center;padding:2rem;border-radius:12px;border:1px solid #e0e0e0"><p style="font-size:11px;color:gray;text-transform:uppercase;letter-spacing:2px">Your wealth survival score</p><p style="font-size:84px;font-weight:700;color:{res["color"]};line-height:1;margin:0">{res["score"]}</p><p style="font-size:16px;color:gray;margin-top:6px">{res["label"]}</p></div>',unsafe_allow_html=True)
    st.markdown('')
    st.info(f'**{country} - {age} - {res["archetype"]}** - Market stress probability: **{res["stress"]}%**. {res["arch_desc"]}')
    st.markdown('#### Signal breakdown')
    for lbl,val in res['signals'].items(): bar(lbl,val)
    st.markdown('---')
    st.markdown(f'**Your investor archetype: {res["archetype"]}**')
    st.markdown(res['arch_desc'])
    st.divider()
    st.markdown('#### Your personalised action plan')
    steps=[]

    # step 1: cash — only show if cash is low
    if ca_val < 15:
        steps.append(('Raise your cash buffer from {}% to at least 20%'.format(ca_val),
            'You are under-protected for a liquidity shock.','+5-6 pts',
            'Your current {}% cash is below the survival threshold. Target 20-25% in short-duration bonds or a money market fund. This is the single highest-impact change you can make.'.format(ca_val)))
    else:
        steps.append(('Maintain your strong cash buffer of {}%'.format(ca_val),
            'Your liquidity position is healthy.','+2 pts',
            'Your {}% cash buffer is above the 20% safety threshold. Keep it here and resist the urge to deploy it all during the next rally.'.format(ca_val)))

    # step 2: commodities — only show rebalance if under or over
    if co_val < 10:
        steps.append(('Add commodity exposure — currently only {}%'.format(co_val),
            'You have almost no inflation or geopolitical hedge.','+4-5 pts',
            'Your {}% commodity allocation leaves you exposed to energy and food shocks. Target 15-20% split between gold ETF (60%) and copper ETF (40%).'.format(co_val)))
    elif co_val > 30:
        steps.append(('Reduce commodity concentration from {}% to 20-22%'.format(co_val),
            'Over-concentration in commodities increases volatility.','-2 pts risk',
            'Your {}% commodities allocation is above optimal. Trim to 20-22% and redeploy the excess into a globally diversified equity ETF.'.format(co_val)))
    else:
        steps.append(('Your commodity allocation of {}% is well positioned'.format(co_val),
            'Good inflation and geopolitical hedge in place.','+2 pts',
            'Ensure your commodity exposure is split between gold (inflation hedge) and copper (industrial cycle exposure) rather than concentrated in one commodity.'))

    # step 3: equities — only show if stocks > 70%
    if s_val > 70:
        steps.append(('Reduce equity concentration from {}% — diversify globally'.format(s_val),
            'Heavy equity concentration amplifies crash exposure.','+3-4 pts',
            'Your {}% equity allocation is high. Shift at least half to an MSCI World ETF to reduce home-country bias and sector concentration. Every 10% you move reduces your crash drawdown by roughly 2-3%.'.format(s_val)))
    elif s_val < 20:
        steps.append(('Consider increasing equity exposure from {}%'.format(s_val),
            'Very low equity allocation limits long-term growth.','+2-3 pts',
            'Your {}% equity allocation may be too conservative for long-term wealth survival. Consider a globally diversified MSCI World ETF for the equity portion.'.format(s_val)))
    else:
        steps.append(('Your equity allocation of {}% is balanced'.format(s_val),
            'Ensure it is globally diversified, not home-country concentrated.','+1-2 pts',
            'Good equity level. Make sure at least 50% is in a global ETF like MSCI World rather than concentrated in your home market index.'))

    # step 4: behavior — always relevant but personalised
    if beh == 'I panic-sold everything':
        steps.append(('Write a panic-prevention rule before the next crash',
            'Your archetype is the highest risk for permanent capital loss.','+4-5 pts',
            'You panic-sold before. Write this sentence now and save it: If my portfolio drops 20%, I will NOT sell. I will buy X amount from my cash buffer instead. Signing a pre-commitment contract with yourself reduces panic-exit probability by 40%.'))
    elif beh == 'I held and watched in pain':
        steps.append(('Convert passive holding into an active rebalancing rule',
            'Watching in pain without a plan leads to eventual capitulation.','+3-4 pts',
            'You held last time — good. But passive holding under stress eventually breaks. Write a rebalancing trigger: if portfolio drops 15%, I buy X from cash. This transforms anxiety into action and protects your recovery.'))
    elif beh == 'I bought the dip':
        steps.append(('Pre-define your dip-buying levels now while markets are calm',
            'Contrarians win only when they execute — not just intend.','+2-3 pts',
            'You bought the dip before — the highest survival archetype. Now pre-define your levels: I will deploy 25% of my cash buffer at -15%, another 25% at -25%, and the final 50% at -35%. Written rules prevent hesitation when the moment arrives.'))
    else:
        steps.append(('Write your pre-commitment rebalancing rule today',
            'Decide your buy levels before the next crash.','+3-4 pts',
            'One sentence: if market falls X%, I invest Y from my cash buffer. Pre-commitment reduces panic-exit frequency by 40%.'))

    # step 5: tail hedge — only if high stress detected
    if res['stress'] > 70:
        steps.append(('Add a tail hedge — market stress is currently at {}%'.format(res['stress']),
            'Current signals justify asymmetric downside protection.','+3-4 pts',
            'With market stress at {}%, a small 3-5% allocation to a put spread on your main index or a volatility ETF gives you insurance precisely when you need it. The psychological permission to hold everything else is as valuable as the payout.'.format(res['stress'])))
    else:
        steps.append(('Monitor tail risk — stress currently moderate at {}%'.format(res['stress']),
            'No immediate hedge urgency but worth watching.','+1-2 pts',
            'Current market stress at {}% does not yet justify an expensive tail hedge. Set a personal alert: if VIX crosses 30, allocate 3-5% to downside protection.'.format(res['stress'])))

    for i,(t,sub,g,d) in enumerate(steps):
        with st.expander(f'Step {i+1}: {t}  |  {g}'): st.markdown(f'*{sub}*'); st.markdown(d)
    sc=res['score']
    scores=[sc,min(sc+6,98),min(sc+11,98),min(sc+15,98),min(sc+20,98),min(sc+24,98)]
    cols=['#E24B4A' if x<55 else '#EF9F27' if x<75 else '#1D9E75' for x in scores]
    fig=go.Figure(go.Bar(x=['Now','Step 1','Step 2','Step 3','Step 4','Step 5'],y=scores,
                         marker_color=cols,text=[str(x) for x in scores],textposition='outside'))
    fig.update_layout(yaxis=dict(range=[0,105]),template='plotly_white',height=300,margin=dict(t=10,b=10))
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Educational tool only. Not financial advice.')
