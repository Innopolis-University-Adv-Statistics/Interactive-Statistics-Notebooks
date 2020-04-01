import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def _plot_distr(distributions: dict, xlim=None, ylim=None):
    """
    Compares multiple distributions and plots their corresponding PDF and CDF
    Params:
    distributions - {name: (scipy.stats.rv_continuous: {kwargs of corresponding distribution})}
    """
    if xlim is None:
        x = np.linspace(
            min([dist.ppf(0.01, **kwargs) for dist, kwargs in distributions.values()]),
            max([dist.ppf(0.99, **kwargs) for dist, kwargs in distributions.values()]),
            num=100,
        )
    else:
        x = np.linspace(xlim[0], xlim[1], num=100)

    pdfs, cdfs, names = zip(
        *[
            (dist.pdf(x, **kwargs), dist.cdf(x, **kwargs), name)
            for name, (dist, kwargs) in distributions.items()
        ]
    )

    pdf_data = pd.DataFrame(np.stack(pdfs, axis=-1), index=x, columns=names)

    cdf_data = pd.DataFrame(np.stack(cdfs, axis=-1), index=x, columns=names)

    f = plt.figure(figsize=(16, 8))
    gs = f.add_gridspec(1, 2)

    pdf_ax = f.add_subplot(gs[0, 0])
    pdf_ax.set_title("PDF")
    if ylim is not None:
        pdf_ax.set_ylim(ylim)
    sns.lineplot(data=pdf_data)

    cdf_ax = f.add_subplot(gs[0, 1])
    cdf_ax.set_title("CDF")
    sns.lineplot(data=cdf_data)
    f.tight_layout()