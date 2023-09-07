import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import scipy.stats as stats
import pyreadstat
import os


def load_prepare_data(outcome='kess'):
    data=[]
    filename=[]

    # Define variables to load from different data files

    variables_acc = ['FCACC_MVPA_MEAN_ACC_E1MIN_100MG']
        
    variables_cog = ['FCWRDSC']

    variables_derived = ['FCCSEX00', 'FDCE0600','FCGTDELAY', 'FCGTDTIME','FCGTQOFDM','FCGTRISKA',
                         'FCGTRISKT','FEMOTION','FCONDUCT','FHYPER','FPEER','FPROSOC','FCBMIN6']
    
    variables_int = ['FCCINE00','FCSPOR00','FCBAND00','FCRJOY00','FCORGA00','FCMUSM00','FCRLSV00','FCPHEX00','FCTVHO00',
                     'FCCOMH00','FCCMEX00','FCINTH00','FCSOME00','FCPLWE00','FCPLWK00','FCROLE00','FCIMWK00','FCPLAB00',
                     'FCBTNG00', 'FCWELK00','FCWHRD00','FCFGHT00','FCSPNT00','FCSTEL00','FCCOPY00','FCHWKM00', 'FCHLPC00', 
                     'FCQTRM00', 'FCSCBE00', 'FCSINT00', 'FCSUNH00', 'FCSTIR00', 'FCSCWA00', 'FCMNWO00', 'FCMISB00', 'FCMISO00', 
                     'FCTRUA00', 'FCATQL00', 'FCSTYY00', 'FCSTYU00', 'FCRELE00', 'FCRELW00', 'FCRELS00', 'FCSCHL00', 'FCVALU00',
                     'FCRELN00', 'FCRLQM00', 'FCRLQF00', 'FCQUAM00', 'FCQUAF00', 'FCOUTW00', 'FCOTWI00', 'FCOTWD00', 'FCDIST00', 
                     'FCDISG00', 'FCDISP00', 'FCCOFA00', 'FCNUFR00', 'FCSTFR0A', 'FCSAFF00', 'FCTRSS00', 'FCNCLS00',
                     'FCBGFR00', 'FCROMG00', 'FCROMB00', 'FCHHND00', 'FCKISS00', 'FCCDDL00', 'FCSMOK00', 'FCALCD00', 
                     'FCALFV00', 'FCCANB00', 'FCOTDR00', 'FCSLLN00', 'FCSLTR00', 'FCSCWK00', 'FCGDSF00',
                     'FCGAMA00', 'FCGMBL00', 'FCGAEM00', 'FCGAMJ00', 'FCBULB00', 'FCBULP00', 'FCHURT00', 'FCPCKP00', 'FCCYBU00',
                     'FCCYBO00', 'FCVICG00', 'FCVICA00', 'FCVICC00', 'FCVICE00', 'FCVICF0A', 'FCHITT00', 'FCWYLK00',
                     'FCWEPN00', 'FCSTLN00', 'FCARES00', 'FCGANG00', 'FCVEGI00', 'FCCGHE00', 'FCWEGT00',
                     'FCEXWT00', 'FCETLS00', 'FCLIFE00', 'FCSATI00', 'FCGDQL00', 'FCDOWL00', 'FCFMLY00',
                     'FCMDSA00', 'FCMDSB00', 'FCMDSC00', 'FCMDSD00', 'FCMDSE00', 'FCFRNS00',
                     'FCMDSF00', 'FCMDSG00', 'FCMDSH00', 'FCMDSI00', 'FCMDSJ00', 'FCMDSK00', 'FCMDSL00', 'FCMDSM00', 'FCHARM00',
                     'FCRISK00', 'FCPTNT00', 'FCTRST0A' ]
    
    variables_fam = ['FDOTHS00','FOEDE000']
    
    variables_par = ['FDKESSL','FDNEUROT','FDAUDIT']
    
    variables_parintcm = ['FPCLSI00', 'FPCLSL00']
    
    variables_parint = ['FPLOLR0G', 'FPSMUS0A']
    
    variables_imde = ['FIMDSCOE']
    
    variables_imdn = ['FIMDSCON']

    variables_imdw = ['FIWIMDSC']
    
    variables_imds = ['FISIMDSC']
    
    variables_weight = ['GOVWT2']
    
    if outcome=='kess':
        variables_out = ['GDCKESSL']
    if outcome == 'mwb':
        variables_out = ['GDWEMWBS']
    if outcome == 'sdq':
        #variables_out = ['GEMOTION_C']
        variables_out = ['GEMOTION']

    categorical=['FDCE0600']
    
    # Inverted items
    
    inverted = ['FCCINE00','FCSPOR00','FCBAND00','FCRJOY00','FCORGA00','FCMUSM00','FCRLSV00','FCPHEX00',
                'FCPLWE00', 'FCPLWK00', 'FCROLE00', 'FCIMWK00', 'FCPLAB00', 'FCBTNG00','FCWELK00','FCWHRD00',
                'FCFGHT00','FCSPNT00','FCSTEL00','FCCOPY00', 'FCHLPC00', 'FCSCBE00', 'FCSINT00', 'FCSUNH00',
                'FCSTIR00', 'FCSCWA00', 'FCMNWO00', 'FCMISB00', 'FCMISO00','FCATQL00', 'FCQUAM00', 'FCQUAF00',
                'FCOUTW00', 'FCOTWI00', 'FCOTWD00', 'FCSTFR0A', 'FCSAFF00', 'FCTRSS00', 'FCNCLS00', 'FCBULB00', 
                'FCBULP00', 'FCHURT00', 'FCPCKP00', 'FCCYBU00', 'FCCYBO00', 'FCCGHE00', 'FCSLTR00','FCVALU00', 
                'FCDOWL00', 'FCGDSF00', 'FCGDQL00', 'FCSATI00', 'FCSCWK00', 'FCWYLK00','FCFMLY00','FCFRNS00',
                'FCSCHL00', 'FCLIFE00','Area_Deprivation']
    
    files = os.listdir('.')

    # Load data files

    for f in files:
        if f[-4:]=='.sav':
            filename.append(f)
            cdata, _=pyreadstat.read_sav(f)
            cdata.columns = [x.upper() for x in cdata.columns]
            data.append(cdata)
    filename=np.array(filename)

    out_idx = np.where(filename=='mcs7_cm_derived.sav')[0][0]
    acc_idx = np.where(filename=='mcs6_cm_accelerometer_derived.sav')[0][0]
    cog_idx=np.where(filename=='mcs6_cm_cognitive_assessment.sav')[0][0]
    derived_idx=np.where(filename=='mcs6_cm_derived.sav')[0][0]
    int_idx=np.where(filename=='mcs6_cm_interview.sav')[0][0]
    fam_idx=np.where(filename=='mcs6_family_derived.sav')[0][0]
    par_idx=np.where(filename=='mcs6_parent_derived.sav')[0][0]
    imde_idx=np.where(filename=='mcs_sweep6_imd_e_2004.sav')[0][0]
    imdn_idx=np.where(filename=='mcs_sweep6_imd_n_2004.sav')[0][0]
    imdw_idx=np.where(filename=='mcs_sweep6_imd_w_2004.sav')[0][0]
    imds_idx=np.where(filename=='mcs_sweep6_imd_s_2004.sav')[0][0]
    parint_idx=np.where(filename=='mcs6_parent_interview.sav')[0][0]
    parintcm_idx=np.where(filename=='mcs6_parent_cm_interview.sav')[0][0]
    weight_idx=np.where(filename=='mcs_longitudinal_family_file.sav')[0][0]

    # Load accelerometer data and average over different assessments if there are more than one
    acc_dat=data[acc_idx][variables_acc+['FCNUM00','MCSID','FCACCAD']].set_index(['MCSID','FCNUM00'])   
    meanaccs=pd.Series(index=acc_dat.index[~acc_dat.index.duplicated()])
    for m in meanaccs.index:
        meanaccs[m]=acc_dat.loc[m,'FCACC_MVPA_MEAN_ACC_E1MIN_100MG'].mean()
    meanaccs.name='Mean_Accelerometer'
    # Load outcome
    out_dat=data[out_idx][variables_out+['GCNUM00','MCSID']].set_index(['MCSID','GCNUM00'])
    out_dat.index.rename('FCNUM00',1,inplace=True)
    # Load derived CM data
    derived_dat=data[derived_idx][variables_derived+['FCNUM00','MCSID']].set_index(['MCSID','FCNUM00'])
    # Load deprivation index data (for all UK countries) and create dummy variable for potential second cohort member of the same family    
    imde_dat1=data[imde_idx][variables_imde+['MCSID']].set_index(['MCSID'])
    imde_dat1['FCNUM00']=1
    imde_dat2 = imde_dat1.copy()
    imde_dat2['FCNUM00']=2
    imde_dat = pd.concat([imde_dat1, imde_dat2], join = 'outer').reset_index().set_index(['MCSID','FCNUM00'])
    imds_dat1=data[imds_idx][variables_imds+['MCSID']].set_index(['MCSID'])
    imds_dat1['FCNUM00']=1
    imds_dat2 = imds_dat1.copy()
    imds_dat2['FCNUM00']=2
    imds_dat = pd.concat([imds_dat1, imds_dat2], join = 'outer').reset_index().set_index(['MCSID','FCNUM00'])
    imdn_dat1=data[imdn_idx][variables_imdn+['MCSID']].set_index(['MCSID'])
    imdn_dat1['FCNUM00']=1
    imdn_dat2 = imdn_dat1.copy()
    imdn_dat2['FCNUM00']=2
    imdn_dat = pd.concat([imdn_dat1, imdn_dat2], join = 'outer').reset_index().set_index(['MCSID','FCNUM00'])
    imdw_dat1=data[imdw_idx][variables_imdw+['MCSID']].set_index(['MCSID'])
    imdw_dat1['FCNUM00']=1
    imdw_dat2 = imdw_dat1.copy()
    imdw_dat2['FCNUM00']=2
    imdw_dat = pd.concat([imdw_dat1, imdw_dat2], join = 'outer').reset_index().set_index(['MCSID','FCNUM00'])
    # Load parent interview about CM data (and use information regardless of source)
    parintcm_dat=data[parintcm_idx][variables_parintcm+['FCNUM00','MCSID']].set_index(['MCSID','FCNUM00']).dropna(how='all')
    # Load parent interview and create table with distinct columns for father and mother and create dummy 
    # variable for potential second cohort member of the same family    
    parint_dat=data[parint_idx][variables_parint+['FPPSEX00','FPCREL00','MCSID']].set_index(['MCSID']).dropna(how='all')
    parint_dat=parint_dat.loc[parint_dat['FPCREL00']==7,:]
    parint_dat['FPSMUS0A']=(parint_dat['FPSMUS0A']==0).astype(int)
    parint_joint=pd.DataFrame(index=parint_dat.index[~parint_dat.index.duplicated()],columns=[x+'f' for x in variables_parint]+[x+'m' for x in variables_parint],data=np.nan)
    for i in parint_joint.index:
        for v in variables_parint:
            try:
                parint_joint.at[i,v+'f']=parint_dat.loc[(parint_dat.index==i)&(parint_dat['FPPSEX00']==1),v]
                parint_joint.at[i,v+'m']=parint_dat.loc[(parint_dat.index==i)&(parint_dat['FPPSEX00']==2),v]
            except:
                pass
    parint_joint['FCNUM00']=1
    parint_joint2 = parint_joint.copy()
    parint_joint2['FCNUM00']=2
    parint_dat = pd.concat([parint_joint, parint_joint2], join = 'outer').reset_index().set_index(['MCSID','FCNUM00'])
    # Load cognitive assessment data
    cog_dat=data[cog_idx][variables_cog+['FCNUM00','MCSID']].set_index(['MCSID','FCNUM00'])
    # Load CM interview data
    int_dat=data[int_idx][variables_int+['FCNUM00','MCSID']].set_index(['MCSID','FCNUM00'])
    # Load derived family data and create dummy variable for potential second cohort member of the same family  
    fam_dat1=data[fam_idx][variables_fam+['MCSID']]
    fam_dat1['FCNUM00']=1
    fam_dat2 = fam_dat1.copy()
    fam_dat2['FCNUM00']=2
    fam_dat = pd.concat([fam_dat1, fam_dat2], join = 'outer').reset_index().drop(columns='index').set_index(['MCSID','FCNUM00'])
    # Load family population weight data and create dummy variable for potential second cohort member of the same family  
    weight_dat1=data[weight_idx][variables_weight+['MCSID']]
    weight_dat1['FCNUM00']=1
    weight_dat2 = weight_dat1.copy()
    weight_dat2['FCNUM00']=2
    weight_dat = pd.concat([weight_dat1, weight_dat2], join = 'outer').reset_index().drop(columns='index').set_index(['MCSID','FCNUM00'])
    # Load derived parent data and create table with distinct columns for father and mother and create dummy 
    # variable for potential second cohort member of the same family    
    par_dat=data[par_idx][variables_par+['FDRES00','MCSID']].set_index(['MCSID']).dropna(how='all')
    par_joint=pd.DataFrame(index=par_dat.index[~par_dat.index.duplicated()],columns=[x+'f' for x in variables_par]+[x+'m' for x in variables_par],data=np.nan)
    for i in par_joint.index:
        for v in variables_par:
            try:
                par_joint.at[i,v+'f']=par_dat.loc[(par_dat.index==i)&(par_dat['FDRES00']==2),v]
                par_joint.at[i,v+'m']=par_dat.loc[(par_dat.index==i)&(par_dat['FDRES00']==1),v]
            except:
                pass
    par_joint['FCNUM00']=1
    par_joint2 = par_joint.copy()
    par_joint2['FCNUM00']=2
    par_dat = pd.concat([par_joint, par_joint2], join = 'outer').reset_index().set_index(['MCSID','FCNUM00'])
    # Merge data files
    joint_dat = derived_dat.join([int_dat, cog_dat, fam_dat, out_dat, par_dat,meanaccs,imde_dat,imds_dat,imdn_dat,imdw_dat,parintcm_dat,parint_dat],how='outer')
    # Combine current and former gang membership
    gangnull=joint_dat['FCGANG00'].isnull()
    joint_dat['FCGANG00']=((joint_dat['FCGANG00']==1)|(joint_dat['FCGANG00']==3)).astype(int)
    joint_dat.loc[gangnull,'FCGANG00']=np.nan
    # Combine religious membership information from different UK countries into one binary religion variable
    religion = pd.Series(index=joint_dat.index,data=True)
    religion.loc[(joint_dat['FCRELE00']==1) | (joint_dat['FCRELW00']==1) | (joint_dat['FCRELS00']==1)| (joint_dat['FCRELN00']==6)]=False
    religion.loc[joint_dat[['FCRELE00','FCRELW00','FCRELS00','FCRELN00']].isnull().all(1)]=np.nan
    joint_dat['Religion']=religion.astype(bool).astype(int)
    joint_dat.drop(columns=['FCRELE00','FCRELW00','FCRELS00','FCRELN00'],inplace=True)
    # Potential sexual minority status if same-sex romantic experience
    romnull=joint_dat[['FCROMG00','FCROMB00']].isnull().any(1)
    sexmin = pd.Series(index=joint_dat.index,data=False)
    sexmin.loc[joint_dat[['FCROMG00','FCROMB00']].sum(1)==2]=True
    sexmin.loc[(joint_dat['FCROMG00']==1)&(joint_dat['FCCSEX00']==0)]=True
    sexmin.loc[(joint_dat['FCROMB00']==1)&(joint_dat['FCCSEX00']==1)]=True
    sexmin.loc[romnull]=np.nan
    joint_dat['PotentialSexMin']=sexmin.astype(float)
    joint_dat.drop(columns=['FCROMG00','FCROMB00'],inplace=True)
    # Combine all gambling behaviors to one binary variable
    joint_dat['Gambling']=(joint_dat[['FCGAMA00', 'FCGMBL00', 'FCGAEM00', 'FCGAMJ00']]==1).any(1)
    joint_dat.loc[joint_dat[['FCGAMA00', 'FCGMBL00', 'FCGAEM00', 'FCGAMJ00']].isnull().any(1)&(joint_dat['Gambling']==False),'Gambling']=np.nan
    joint_dat['Gambling']=joint_dat['Gambling'].astype(float)
    joint_dat.drop(columns=['FCGAMA00', 'FCGMBL00', 'FCGAEM00', 'FCGAMJ00'],inplace=True)
    # Combine are deprivation index for different UK countries
    joint_dat['Area_Deprivation']=joint_dat[['FIMDSCOE','FISIMDSC', 'FIMDSCON', 'FIWIMDSC']].sum(1)
    joint_dat.drop(columns=['FIMDSCOE','FISIMDSC', 'FIMDSCON', 'FIWIMDSC'],inplace=True)
    # Create impairment by chronic disease variable (including 0 = no impairment)
    joint_dat.loc[joint_dat['FPCLSI00']==2,'ChronicDisease']=0
    joint_dat.loc[joint_dat['FPCLSI00']==1,'ChronicDisease']=4-joint_dat.loc[joint_dat['FPCLSI00']==1,'FPCLSL00']
    # joint_dat['ChronicDisease']=((joint_dat['FPCLSI00']==1)&(joint_dat['FPCLSL00']<3)).astype(int)
    joint_dat.drop(columns=['FPCLSI00','FPCLSL00'],inplace=True)
    # Code binary variables as 0,1 instead of 2,1
    binary=joint_dat.columns[joint_dat.nunique()==2]
    joint_dat[binary]=joint_dat[binary].replace(2,0)
    # # Dropped cases: 350 due to more than 80% missing predictors, rest dummy participants
    # dropped_c = joint_dat.drop(index=joint_dat.index[joint_dat.isnull().mean(axis=1)<=0.8])
    # Remove dummy cases (potential second cohort members of the same family) and cases with extremely incomplete data
    joint_dat.drop(index=joint_dat.index[joint_dat.isnull().mean(axis=1)>0.8],inplace=True)

    # Select cases with available outcome data
    y=joint_dat[variables_out].dropna()
    X = joint_dat.loc[y.index,:].drop(columns=variables_out)
    # Invert inverted items
    X[inverted] = -X[inverted]
    # Get population weights
    weight=y.join(weight_dat).iloc[:,-1]

    return X, y.iloc[:,0], categorical, weight    