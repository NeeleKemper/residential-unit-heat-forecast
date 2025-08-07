# residential-unit-heat-forecast
## Forecasting of Residential Unit's Heat Demands - A Comparison of Machine Learning Techniques in a Real-world Case Study
A large proportion of the energy consumed by private households is used for space heating and domestic hot water. In the context of the energy transition, the predominant aim is to reduce this consumption. In addition to implementing better energy standards in new buildings and refurbishing old buildings, intelligent energy management concepts can also contribute by operating heat generators according to demand based on an expected heat requirement. This requires forecasting models for heat demand to be as accurate and reliable as possible. In this paper, we present a case study of a newly built medium-sized living quarter in central Europe made up of 66 residential units from which we gathered consumption data for almost two years. Based on this data, we investigate the possibility of forecasting heat demand using a variety of time series models and offline and online machine learning (ML) techniques in a standard data science approach. We chose to analyze different modeling techniques as they can be used in different settings, where time series models require no additional data, offline ML needs a lot of data gathered up front, and online ML could be deployed from day one. A special focus lies on peak demand and outlier forecasting, as well as investigations into seasonal expert models. We also highlight the computational expense and explainability characteristics of the used models. We compare the used methods with naive models as well as each other, finding that time series models, as well as online ML, do not yield promising results. Accordingly, we will deploy one of the offline ML models in our real-world energy management system in the near future.


# Citing the Project
```
@article{kemper2025forecasting,
  title={Forecasting of residential unit’s heat demands: a comparison of machine learning techniques in a real-world case study},
  author={Kemper, Neele and Heider, Michael and Pietruschka, Dirk and H{\"a}hner, J{\"o}rg},
  journal={Energy Systems},
  volume={16},
  number={1},
  pages={281--315},
  year={2025},
  publisher={Springer}
}
```
