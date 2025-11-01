import pandas as pd
import numpy as np
import os

# Work based on "Observational before-after studies in road safety" by Ezra Hauer (2002)
# Functions
def NaiveBeforeAfter(
    df_treatment=None,
    before_duration="treatment_before_duration",
    after_duration="treatment_after_duration",
    before_count="treatment_before_count",
    after_count="treatment_after_count",
    ):

    """
    Calculates the Naive BA by the 4-step expanded

    Based on:
    Chapter 7, "The Naive before-after study"
    Chapter 10, "The variability of treatment effect"
    """

    # Copy dataset
    df_treatment = df_treatment.copy()

    # For composite entities
    df_treatment["rdj"] = df_treatment[after_duration]/df_treatment[before_duration]
    df_treatment["rdj_x_before_count"] = df_treatment["rdj"]*df_treatment[before_count]
    df_treatment["rdj2_x_before_count"] = df_treatment["rdj"]*df_treatment["rdj_x_before_count"]

    # Step 1
    lambda_par = sum(df_treatment[after_count]) # estimated after with treatment

    pi_par = sum(df_treatment["rdj_x_before_count"])# predict without treatment

    # Step 2 
    var_lambda_par = sum(df_treatment[after_count]) # assumed to be Poisson distributed
    std_lambda_par = var_lambda_par**0.5

    var_pi_par = sum(df_treatment["rdj2_x_before_count"]) # assumed to be Poisson distributed
    std_pi_par = var_pi_par**0.5

    # Step 3
    delta_par = pi_par - lambda_par
    delta_norm_par = delta_par/sum(df_treatment[after_duration])

    teta_par = (lambda_par/pi_par) / (1+(var_pi_par/(pi_par**2)))

    # Step 4
    var_delta_par = var_lambda_par + var_pi_par
    std_delta_par = var_delta_par**0.5

    var_teta_par = (teta_par**2)*((var_lambda_par/(lambda_par**2))+(var_pi_par/(pi_par**2)))/((1+(var_pi_par/(pi_par**2)))**2)
    std_teta_par = var_teta_par**0.5

    # For single entities (variability of treatment effect)
    df_treatment["lambda_par"] = df_treatment[after_count]
    df_treatment["var_lambda_par"] = df_treatment[after_count]
    df_treatment["pi_par"] = df_treatment["rdj_x_before_count"]
    df_treatment["var_pi_par"] = df_treatment["rdj2_x_before_count"]

    df_treatment,s2_teta,avg_V,var_teta_vte,std_teta_vte = VariabilityTreatmentEffect(
        df_treatment,
        lambda_par="lambda_par",
        var_lambda_par="var_lambda_par",
        pi_par="pi_par",
        var_pi_par="var_pi_par",
        )

    result = {
        "lambda":lambda_par,
        "var_lambda":var_lambda_par,
        "std_lambda":std_lambda_par,
        "pi":pi_par,
        "var_pi":var_pi_par,
        "std_pi":std_pi_par,
        "delta":delta_par,
        "delta_norm":delta_norm_par,
        "var_delta":var_delta_par,
        "std_delta":std_delta_par,
        "teta":teta_par,
        "var_teta":var_teta_par,
        "std_teta":std_teta_par,
        "s2_teta":s2_teta,
        "avg_V":avg_V,
        "var_teta_vte":var_teta_vte,
        "std_teta_vte":std_teta_vte
      }
    result = pd.DataFrame(pd.Series(result,name="NAive BA"))

    df_treatment = df_treatment[[
        before_duration,after_duration,
        before_count,after_count,
        "rdj","rdj_x_before_count","rdj2_x_before_count",
        "lambda_par","var_lambda_par","pi_par","var_pi_par",
        "teta_par","var_teta_par","std_teta_par"
        ]]

    return df_treatment,result

def ComparisonGroupBeforeAfter(
    df_treatment=None,
    df_comparison_group=None,
    treatment_before_duration="treatment_before_duration",
    treatment_after_duration="treatment_after_duration",
    treatment_before_count="treatment_before_count",
    treatment_after_count="treatment_after_count",
    comparison_group_before_duration="comparison_group_before_duration",
    comparison_group_after_duration="comparison_group_after_duration",
    comparison_group_before_count="comparison_group_before_count",
    comparison_group_after_count="comparison_group_after_count",
    var_w_par=0.001):

    # Copy dataset
    df_treatment = df_treatment.copy()
    df_comparison_group = df_comparison_group.copy()

    df_treatment["rdj"] = df_treatment[treatment_after_duration]/df_treatment[treatment_before_duration]
    df_treatment["rdj_x_before_count"] = df_treatment["rdj"]*df_treatment[treatment_before_count]
    df_treatment["rdj2_x_before_count"] = df_treatment["rdj"]*df_treatment["rdj_x_before_count"]

    df_comparison_group["rdj"] = df_comparison_group[comparison_group_after_duration]/df_comparison_group[comparison_group_before_duration]
    df_comparison_group["rdj_x_before_count"] = df_comparison_group["rdj"]*df_comparison_group[comparison_group_before_count]
    df_comparison_group["rdj2_x_before_count"] = df_comparison_group["rdj"]*df_comparison_group["rdj_x_before_count"]

    L = sum(df_treatment[treatment_after_count]) # estimated after with treatment
    K = sum(df_treatment[treatment_before_count])# predict without treatment (brefore, NaiveBA)
    M = sum(df_comparison_group[comparison_group_before_count]) # estimated before G-C
    N = sum(df_comparison_group[comparison_group_after_count]) # estimated after G-C
    
    # Step 1
    lambda_par = L
    rt = (N/M) / (1+(1/M))
    pi_par = rt*K
    
    # Step 2
    var_lambda_par = L
    std_lambda_par = var_lambda_par**0.5

    var_rt_par = (1/M) + (1/N) + var_w_par

    var_pi_par = (pi_par**2)*((1/K)+var_rt_par)
    std_pi_par = var_pi_par**0.5

    # Step 3
    delta_par = pi_par - lambda_par
    delta_norm_par = delta_par/sum(df_treatment[treatment_after_duration])

    teta_par = (lambda_par/pi_par) / (1+(var_pi_par/(pi_par**2)))

    # Step 4
    var_delta_par = var_lambda_par + var_pi_par
    std_delta_par = var_delta_par**0.5

    var_teta_par = (teta_par**2)*((var_lambda_par/(lambda_par**2))+(var_pi_par/(pi_par**2)))/((1+(var_pi_par/(pi_par**2)))**2)
    std_teta_par = var_teta_par**0.5

    # For single entities (variability of treatment effect)
    df_treatment["lambda_par"] = df_treatment[treatment_after_count]
    df_treatment["var_lambda_par"] = df_treatment[treatment_after_count]
    df_treatment["pi_par"] = df_treatment["rdj_x_before_count"]
    df_treatment["var_pi_par"] = df_treatment["rdj2_x_before_count"]

    df_treatment,s2_teta,avg_V,var_teta_vte,std_teta_vte = VariabilityTreatmentEffect(
        df_treatment,
        lambda_par="lambda_par",
        var_lambda_par="var_lambda_par",
        pi_par="pi_par",
        var_pi_par="var_pi_par",
        )

    result = {
        "K":K,
        "L":L,
        "M":M,
        "N":N,
        "rt":rt,
        "var_rt/rt2":var_rt_par,
        "var_w":var_w_par,
        "lambda":lambda_par,
        "var_lambda":var_lambda_par,
        "std_lambda":std_lambda_par,
        "pi":pi_par,
        "var_pi":var_pi_par,
        "std_pi":std_pi_par,
        "delta":delta_par,
        "delta_norm":delta_norm_par,
        "var_delta":var_delta_par,
        "std_delta":std_delta_par,
        "teta":teta_par,
        "var_teta":var_teta_par,
        "std_teta":std_teta_par,
        "s2_teta":s2_teta,
        "avg_V":avg_V,
        "var_teta_vte":var_teta_vte,
        "std_teta_vte":std_teta_vte
      }
    
    result = pd.DataFrame(pd.Series(result,name="CG BA"))

    df_treatment = df_treatment[[
        treatment_before_duration,
        treatment_after_duration,
        treatment_before_count,
        treatment_after_count,
        "rdj","rdj_x_before_count","rdj2_x_before_count",
        "teta_par","var_teta_par","std_teta_par"]]

    df_comparison_group = df_comparison_group[[
        comparison_group_before_duration,
        comparison_group_after_duration,
        comparison_group_before_count,
        comparison_group_after_count,
        "rdj","rdj_x_before_count","rdj2_x_before_count"]]

    return df_treatment,df_comparison_group,result

def VariabilityTreatmentEffect(
        df_treatment,
        lambda_par="lambda_par",
        var_lambda_par="var_lambda_par",
        pi_par="pi_par",
        var_pi_par="var_pi_par",
):
    """
    Based on Chapter 10, "The variability of treatment effect"
    """
    # Copy dataset
    df_treatment = df_treatment.copy()

    # Teta by entity
    df_treatment["teta_par"] = (df_treatment[lambda_par]/df_treatment[pi_par]) / (1+(df_treatment[var_pi_par]/(df_treatment[pi_par]**2)))
    df_treatment["var_teta_par"] = (df_treatment["teta_par"]**2)*((df_treatment[var_lambda_par]/(df_treatment[lambda_par]**2))+(df_treatment[var_pi_par]/(df_treatment[pi_par]**2)))/((1+(df_treatment[var_pi_par]/(df_treatment[pi_par]**2)))**2)
    df_treatment["std_teta_par"] = np.sqrt(df_treatment["var_teta_par"])

    # Variability of Treatment Effect (vte)
    s2_teta = np.var(df_treatment["teta_par"],ddof=1)
    avg_V = np.mean(df_treatment["var_teta_par"])
    var_teta_vte = s2_teta - avg_V
    std_teta_vte = np.sqrt(var_teta_vte)

    return df_treatment,s2_teta,avg_V,var_teta_vte,std_teta_vte

def EmpiricalBayesMethod(
    df_treatment,
    SPF_func,
    b,
    par_list,
    start_year,
    end_year,
    before_duration="treatment_before_duration",
    after_duration="treatment_after_duration",
    before_count="treatment_before_count",
    after_count="treatment_after_count",
    sep="@"):

    """
    The "df_treatment should contain the combination of the "par_list" and range years columns with "sep" as a separator.
    Ex.: "AADV@2025"
    """

    df_treatment = df_treatment.copy()

    # MEB
    accident_before_SPF_cols= []
    accident_after_SPF_cols = []
    
    for y in range(start_year,end_year+1):
        df_treatment[f"accident_func_SPF"+sep+str(y)] = df_treatment.apply(lambda row:SPF_func([y]+[row[par+sep+str(y)] for par in par_list]),axis=1)

        df_treatment["accident_before_SPF"+sep+str(y)] = df_treatment[before_duration+sep+str(y)] * df_treatment[f"accident_func_SPF"+sep+str(y)]
        accident_before_SPF_cols.append("accident_before_SPF"+sep+str(y))

        df_treatment["accident_after_SPF"+sep+str(y)] = df_treatment[after_duration+sep+str(y)] * df_treatment[f"accident_func_SPF"+sep+str(y)]
        accident_after_SPF_cols.append("accident_after_SPF"+sep+str(y))

    df_treatment["Ekb SPF"] = df_treatment[accident_before_SPF_cols].sum(axis=1)
    df_treatment["Eka SPF"] = df_treatment[accident_after_SPF_cols].sum(axis=1)
    
    df_treatment["VAR_Ek SPF"] = (df_treatment["Ekb SPF"]**2)/b
    
    df_treatment["alpha"] = 1/(1+(df_treatment["VAR_Ek SPF"]/df_treatment["Ekb SPF"]))
    
    df_treatment["k_par"] = (df_treatment["alpha"]*df_treatment["Ekb SPF"])+((1-df_treatment["alpha"])*df_treatment[before_count])
    df_treatment["var_k_par"] = (1-df_treatment["alpha"])*df_treatment["k_par"]
    
    df_treatment["rC"] = df_treatment["Eka SPF"]/df_treatment["Ekb SPF"]
    df_treatment["pi_par"] = df_treatment["rC"]*df_treatment["k_par"]
    df_treatment["var_pi_par"] = (df_treatment["rC"]**2)*df_treatment["var_k_par"]

    # 4-steps
    # Step 1
    lambda_par = sum(df_treatment[after_count]) # estimated after with treatment
    
    pi_par = sum(df_treatment["pi_par"])# predict without treatment
    
    # Step 2 
    var_lambda_par = sum(df_treatment[after_count]) # assumed to be Poisson distributed
    std_lambda_par = var_lambda_par**0.5
    
    var_pi_par = sum(df_treatment["var_pi_par"])
    std_pi_par = var_pi_par**0.5
    
    # Step 3
    delta_par = pi_par - lambda_par
    delta_norm_par = delta_par/sum(df_treatment[after_duration])
    
    teta_par = (lambda_par/pi_par) / (1+(var_pi_par/(pi_par**2)))
    
    # Step 4
    var_delta_par = var_lambda_par + var_pi_par
    std_delta_par = var_delta_par**0.5
    
    var_teta_par = (teta_par**2)*((var_lambda_par/(lambda_par**2))+(var_pi_par/(pi_par**2)))/((1+(var_pi_par/(pi_par**2)))**2)
    std_teta_par = var_teta_par**0.5
    
    # For single entities (variability of treatment effect)
    df_treatment["lambda_par"] = df_treatment[after_count]
    df_treatment["var_lambda_par"] = df_treatment[after_count]
    
    df_treatment,s2_teta,avg_V,var_teta_vte,std_teta_vte = VariabilityTreatmentEffect(
        df_treatment,
        lambda_par="lambda_par",
        var_lambda_par="var_lambda_par",
        pi_par="pi_par",
        var_pi_par="var_pi_par",
        )
    
    result = {
        "lambda":lambda_par,
        "var_lambda":var_lambda_par,
        "std_lambda":std_lambda_par,
        "pi":pi_par,
        "var_pi":var_pi_par,
        "std_pi":std_pi_par,
        "delta":delta_par,
        "delta_norm":delta_norm_par,
        "var_delta":var_delta_par,
        "std_delta":std_delta_par,
        "teta":teta_par,
        "var_teta":var_teta_par,
        "std_teta":std_teta_par,
        "s2_teta":s2_teta,
        "avg_V":avg_V,
        "var_teta_vte":var_teta_vte,
        "std_teta_vte":std_teta_vte
      }
    result = pd.DataFrame(pd.Series(result,name="EB"))
    
    return df_treatment,result

if __name__=="__main__":
    print("Ok")
    