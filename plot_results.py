import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle

files = glob.glob(os.path.join('CIA_results/pkl','*.pkl'))
alpha_values = np.linspace(0.01, 0.1, 10)

for file in files:
    data_name = file.split('/')[-1].split('.pkl')[0]
    with open(file, 'rb') as f:
        results = pickle.load(f)
        
    coverage_abs, efficiency_abs, coverage_quantile, efficiency_quantile, \
        coverage_normal_iqr, efficiency_normal_iqr, coverage_normal_conformal, efficiency_normal_conformal, \
        coverage_abs_stratified, efficiency_abs_stratified, coverage_quantile_stratified, efficiency_quantile_stratified, \
        coverage_abs_stratified_null, efficiency_abs_stratified_null, coverage_quantile_stratified_null, efficiency_quantile_stratified_null, \
        coverage_abs_bonferroni, efficiency_abs_bonferroni, coverage_quantile_bonferroni, efficiency_quantile_bonferroni = results

    methods = [
        "CIA (Split)",
        "CIA (CQR)",
        "CIA (Split) Stratified",
        "CIA (CQR) Stratified",
        "Group (Split)",
        "Group (CQR)",
        "Normal (Hetero)",
        "Normal (Homo)",
        "Bonferroni (Split)",
        "Bonferroni (CQR)"
    ]

    coverage_data = [
        coverage_abs,
        coverage_quantile,
        coverage_abs_stratified,
        coverage_quantile_stratified,
        coverage_abs_stratified_null,
        coverage_quantile_stratified_null,
        coverage_normal_iqr,
        coverage_normal_conformal,
        coverage_abs_bonferroni,
        coverage_quantile_bonferroni
    ]

    efficiency_data = [
        efficiency_abs,
        efficiency_quantile,
        efficiency_abs_stratified,
        efficiency_quantile_stratified,
        efficiency_abs_stratified_null,
        efficiency_quantile_stratified_null,
        efficiency_normal_iqr,
        efficiency_normal_conformal,
        efficiency_abs_bonferroni,
        efficiency_quantile_bonferroni
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    for i in range(len(methods)):
        ax1.errorbar(alpha_values, coverage_data[i].mean(axis=0), yerr=coverage_data[i].std(axis=0),
                     label=methods[i], linewidth=2,
                     markersize=8, marker='o')
        ax2.plot(coverage_data[i].mean(axis=0), efficiency_data[i].mean(axis=0), linewidth=2,
                 markersize=8, marker='o')

    ax1.plot(alpha_values, 1 - alpha_values, '--', color='red', linewidth=2)
    ax1.set_xlabel(r'$ \alpha $', fontsize=22)
    ax1.set_ylabel('Coverage', fontsize=22)
    ax1.tick_params(axis='both', labelsize=14)
    ax2.set_xlabel('Coverage', fontsize=22)
    ax2.set_ylabel('Size', fontsize=22)
    ax2.tick_params(axis='both', labelsize=14)

    fig.suptitle(data_name.replace('_', ''), fontsize=30)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='lower center', ncol=3, fontsize=20)

    plt.tight_layout(rect=[0, 0.22, 1, 0.99])  # Adjust the spacing to accommodate the legend
    plt.show()
    # fig.savefig(f"CIA_results/appendix/{data_name}.pdf", dpi=300)

    methods = [
        "CIA",
        "Group",
        "Normal",
        "Bonferroni"
    ]

    coverage_data = [
        coverage_quantile_stratified,
        coverage_quantile_stratified_null,
        coverage_normal_iqr,
        coverage_quantile_bonferroni
    ]

    efficiency_data = [
        efficiency_quantile_stratified,
        efficiency_quantile_stratified_null,
        efficiency_normal_iqr,
        efficiency_quantile_bonferroni
    ]

    colors = ['red', 'blue', 'green', 'orange']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    for i in range(len(methods)):
        ax1.errorbar(alpha_values, coverage_data[i].mean(axis=0), yerr=coverage_data[i].std(axis=0),
                     label=methods[i], linewidth=2, color=colors[i],
                     markersize=8, marker='o')
        ax2.plot(coverage_data[i].mean(axis=0),
                 efficiency_data[i].mean(axis=0),
                 linewidth=2, color=colors[i], markersize=8, marker='o')

    ax1.plot(alpha_values, 1 - alpha_values, '--', color='black', linewidth=2)
    ax1.set_xlabel(r'$ \alpha $', fontsize=24)
    ax1.set_ylabel('Coverage', fontsize=24)
    ax1.tick_params(axis='both', labelsize=18)
    ax2.set_xlabel('Coverage', fontsize=24)
    ax2.set_ylabel('Size', fontsize=24)
    ax2.tick_params(axis='both', labelsize=18)

    fig.suptitle(data_name.replace('_', ''), fontsize=30)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='lower center', ncol=4, fontsize=20)

    plt.tight_layout(rect=[0, 0.1, 1, 0.99])  # Adjust the spacing to accommodate the legend
    plt.show()
    # fig.savefig(f"CIA_results/main/{data_name}.pdf", dpi=300)
