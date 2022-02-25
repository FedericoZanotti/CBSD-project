import bnlearn as bn

def get_merged_df(df1, df2):

  df1_changed = df1.rename(columns={'Pr1': 'Pr1_D', 'Pr2': 'Pr2_D', 'Pr3':'Pr3_D','FA1':'FA1_D',
                                                      'FA2':'FA2_D','FA3':'FA3_D',
                                                      'FI1':'FI1_D', 'FI2':'FI2_D', 'FI3':'FI3_D'})
  df1_changed=df1_changed.reset_index(drop=True)
  new_df=df2.reset_index(drop=True)
  new_df=new_df.join(df1_changed)
  return new_df

def generate_samples(df, samples=1000):
  res_dic={}
  model_hc_k2   = bn.structure_learning.fit(df, methodtype='hc', scoretype='k2')
  model_hc_bdeu = bn.structure_learning.fit(df, methodtype='hc', scoretype='bdeu')
  model_cl_k2   = bn.structure_learning.fit(df, methodtype='cl', scoretype='k2')
  model_cl_bdeu = bn.structure_learning.fit(df, methodtype='cl', scoretype='bdeu')
  model_cl_bic  = bn.structure_learning.fit(df, methodtype='cl', scoretype='bic')

  hc_k2 = bn.parameter_learning.fit(model_hc_k2,df, methodtype='ml', verbose=0)
  hc_bdeu = bn.parameter_learning.fit(model_hc_bdeu,df, methodtype='ml', verbose=0)
  cl_bic = bn.parameter_learning.fit(model_cl_bic,df, methodtype='ml', verbose=0)
  cl_k2 = bn.parameter_learning.fit(model_cl_k2,df, methodtype='ml', verbose=0)
  cl_bdeu = bn.parameter_learning.fit(model_cl_bdeu,df, methodtype='ml', verbose=0)

  all_generation_k2 = bn.sampling(hc_k2, n=samples)
  all_generation_bdeu = bn.sampling(hc_bdeu, n=samples)
  all_generation_cl_bic = bn.sampling(cl_bic, n=samples)
  all_generation_cl_k2 = bn.sampling(cl_k2, n=samples)
  all_generation_cl_bdeu = bn.sampling(cl_bdeu, n=samples)


  all_generation_k2=all_generation_k2.reindex(columns=df.columns)
  all_generation_bdeu=all_generation_bdeu.reindex(columns=df.columns)
  all_generation_cl_bic=all_generation_cl_bic.reindex(columns=df.columns)
  all_generation_cl_k2=all_generation_cl_k2.reindex(columns=df.columns)
  all_generation_cl_bdeu=all_generation_cl_bdeu.reindex(columns=df.columns)
  res_dic['hc_k2']=all_generation_k2
  res_dic['hc_bdeu']=all_generation_bdeu
  res_dic['cl_bic']=all_generation_cl_bic
  res_dic['cl_k2']=all_generation_cl_k2
  res_dic['cl_bdeu']=all_generation_cl_bdeu


  return res_dic