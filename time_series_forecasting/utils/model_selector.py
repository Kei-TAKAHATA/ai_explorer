from time_series_forecasting.models.local_seasonal_model import LocalSeasonalModel


class ModelSelector:
    def __init__(self, settings):
        self.settings = settings["model"]

    def get_model(self):
        model_name = self.settings["model_name"]
        if model_name == "local_seasonal_model":
            return LocalSeasonalModel(
                seasonal_periods=self.settings["model_params"]["seasonal_periods"],
                include_trend=self.settings["model_params"]["include_trend"],
            )
        raise ValueError(f"Model {model_name} not found")
