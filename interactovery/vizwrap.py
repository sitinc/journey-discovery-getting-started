# MIT License
#
# Copyright (c) 2024, Justin Randall, Smart Interactive Transformations Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import matplotlib.pyplot as plt


class MetricChart:
    """
    Metric count class.
    """

    def __init__(self,
                 *,
                 title: str,
                 legend_title: str = None,
                 metrics: list[str],
                 counts: list[int],
                 ):
        """
        Construct a new instance.

        :param title: The chart title.
        :param legend_title: The legend title.
        :param metrics: The metrics list data.
        :param counts: The counts list data.
        """
        self.title = title
        self.legend_title = legend_title
        self.metrics = metrics
        self.counts = counts

    def __str__(self):
        return (f"MetricChart(title={self.title}" +
                f", legend_title={self.legend_title}" +
                f", metrics={self.metrics}" +
                f", counts={self.counts}" +
                ")")

    def __repr__(self):
        return (f"MetricChart(title={self.title!r}" +
                f", legend_title={self.legend_title!r}" +
                f", metrics={self.metrics!r}" +
                f", counts={self.counts!r}" +
                ")")

    def clone(self,
              *,
              title: str = None,
              legend_title: str = None,
              metrics: list[str] = None,
              counts: list[int] = None,
              ):
        return MetricChart(
            title=title if title is not None else self.title,
            legend_title=legend_title if legend_title is not None else self.legend_title,
            metrics=metrics if metrics is not None else self.metrics,
            counts=counts if counts is not None else self.counts,
        )

    def group(self, threshold: float = None, **kwargs):
        metric_names = self.metrics
        metric_counts = self.counts

        if threshold is None:
            raise Exception("threshold is required")

        # Group slices below 1%
        total_metric_counts = sum(metric_counts)
        threshold = threshold * total_metric_counts
        small_metric_names = [count for count in metric_counts if count < threshold]
        other_count = sum(small_metric_names)

        # Filter out small files and add "Other" category
        metric_counts_filtered = [count for count in metric_counts if count >= threshold]
        metric_names_filtered = [metric_names[i] for i, count in enumerate(metric_counts) if count >= threshold]

        if other_count > 0:
            metric_counts_filtered.append(other_count)
            metric_names_filtered.append("Other")

        final_chart = MetricChart(
            metrics=metric_names_filtered,
            counts=metric_counts_filtered,
            **kwargs,
        )
        return final_chart


class VizWrap:
    """
    Utility class for visualizing interactional data.
    """

    def __init__(self,
                 *,
                 debug: bool = None,
                 ):
        """
        Construct a new instance.

        :param debug: The debug flag.
        """
        self.debug = debug

    @staticmethod
    def show_pie(*,
                 chart: MetricChart,
                 sort_dsc: bool = True,
                 group_threshold: float = None,
                 ):
        if group_threshold is not None:
            group_threshold_percent = group_threshold * 100
            final_chart = chart.group(
                threshold=group_threshold,
                title=f'{chart.title} (<{group_threshold_percent}% as Other)',
                legend_title=chart.legend_title,
            )
        else:
            final_chart = chart.clone(
                title=chart.title,
                legend_title=chart.legend_title,
            )

        metric_names = final_chart.metrics
        metric_counts = final_chart.counts
        title = final_chart.title
        legend_title = final_chart.legend_title

        # Sort the slices by size
        sorted_data = sorted(zip(metric_counts, metric_names), reverse=sort_dsc)
        metric_counts_sorted, metric_names_sorted = zip(*sorted_data)

        # Adjust the figure size
        plt.figure(figsize=(12, 12))

        wedges, texts, autotexts = plt.pie(metric_counts_sorted, autopct='%1.1f%%', startangle=140)

        # Equal aspect ratio ensures the pie chart is circular.
        plt.axis('equal')

        # Add a legend
        plt.legend(wedges, metric_names_sorted, title=legend_title, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        plt.title(title)
        plt.show()

    @staticmethod
    def show_horiz_bars(*,
                        chart: MetricChart,
                        sort_dsc: bool = True,
                        ):
        metric_names = chart.metrics
        metric_counts = chart.counts
        title = chart.title
        legend_title = chart.legend_title

        # Sort the files by line count in descending order
        sorted_data = sorted(zip(metric_counts, metric_names), reverse=sort_dsc)
        metric_counts_sorted, metric_names_sorted = zip(*sorted_data)

        # Adjust the figure size to accommodate all entries
        plt.figure(figsize=(10, len(metric_names_sorted) * 0.5))

        plt.barh(metric_names_sorted, metric_counts_sorted)
        plt.xlabel(legend_title)
        plt.title(title)
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest value on top
        plt.tight_layout()  # Adjust layout to fit all labels
        plt.show()

    @staticmethod
    def visualize_intent_bars(*,
                              chart: MetricChart,
                              sort_dsc: bool = True,
                              ):
        final_chart = MetricChart(
            title=f'Number of {chart.title} per-Cluster (Sorted)',
            legend_title='Utterance Count',
            metrics=chart.metrics,
            counts=chart.counts,
        )
        VizWrap.show_horiz_bars(chart=final_chart, sort_dsc=sort_dsc)

    @staticmethod
    def visualize_intent_pie(*,
                             chart: MetricChart,
                             sort_dsc: bool = True,
                             group_threshold: float = None,
                             ):
        final_chart = chart.clone(
            title=f'Distribution of {chart.title} by Clustered Intent',
            legend_title='Intents',
        )
        VizWrap.show_pie(chart=final_chart, sort_dsc=sort_dsc, group_threshold=group_threshold)

    @staticmethod
    def visualize_entity_bars(*,
                              chart: MetricChart,
                              sort_dsc: bool = True,
                              ):
        final_chart = chart.clone(
            title='Number of Values per-Entity (Sorted)',
            legend_title='Value Count',
        )
        VizWrap.show_horiz_bars(chart=final_chart, sort_dsc=sort_dsc)

    @staticmethod
    def visualize_entity_pie(*,
                             chart: MetricChart,
                             sort_dsc: bool = True,
                             group_threshold: float = None,
                             ):
        final_chart = chart.clone(
            title='Distribution of Values by Entity Type (Sorted by Size)',
            legend_title='Entities',
        )
        VizWrap.show_pie(chart=final_chart, sort_dsc=sort_dsc, group_threshold=group_threshold)
