import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


class PlotConfig:
    # TODO: update to actual width
    WIDTH_INCHES = 5.5
    MAJOR_FONT_SIZE = 8
    MINOR_FONT_SIZE = 6

    @classmethod
    def setup(cls):
        sns.set(style="whitegrid")
        mpl.rc("text", usetex=True)
        plt.rcParams["text.usetex"] = True
        # plt.rcParams["text.latex.preamble"] = r""
        plt.rcParams["font.size"] = cls.MAJOR_FONT_SIZE
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Times New Roman"
        plt.rcParams["axes.labelsize"] = cls.MAJOR_FONT_SIZE
        plt.rcParams["xtick.labelsize"] = cls.MINOR_FONT_SIZE
        plt.rcParams["ytick.labelsize"] = cls.MINOR_FONT_SIZE
        plt.rcParams['axes.titlesize'] = cls.MAJOR_FONT_SIZE
        plt.rcParams['legend.fontsize'] = cls.MINOR_FONT_SIZE 


    @classmethod
    def convert_width(
        cls, fsize: tuple[float, float], page_scale: float = 0.5
    ) -> tuple[float, float]:
        """converts an arbitrary figure size into an appropriate figure size maintaining
        the aspect ration."""
        rescale_width = cls.WIDTH_INCHES * page_scale
        width = fsize[0]
        return tuple(size * rescale_width / width for size in fsize)


    @classmethod
    def save_fig(cls, fig, name: str):
        """Saves a figure with the appropriate size and resolution."""
        name = f"{name}.pdf"
        fig.savefig(name, format="pdf", bbox_inches="tight")
