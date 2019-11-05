.. mdinclude:: landing.md


.. toctree::
  :maxdepth: 1
  :hidden:

  Home <self>


.. toctree::
  :maxdepth: 1
  :caption: Overview

  overview/getting_started.md


.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/overview.md
   examples/train_and_evaluate.ipynb
   examples/trading_context.ipynb


.. toctree::
  :maxdepth: 1
  :caption: Component Types

  components/components.md
  components/component_types/trading_environment.md
  components/component_types/instrument_exchange.md
  components/component_types/feature_pipeline.md
  components/component_types/action_strategy.md
  components/component_types/reward_strategy.md
  components/component_types/trading_strategy.md


.. toctree::
  :maxdepth: 1
  :caption: Trading Environment

  components/environments/TradingEnvironment.md


.. toctree::
  :maxdepth: 1
  :caption: Exchanges
  
  components/exchanges/InstrumentExchange.md
  components/exchanges/simulated/SimulatedExchange.md
  components/exchanges/simulated/FBMExchange.md
  components/exchanges/live/CCXTExchange.md
  components/exchanges/live/InteractiveBrokersExchange.md


.. toctree::
  :maxdepth: 1
  :caption: Features


  components/features/FeaturePipeline.md
  components/features/FeatureTransformer.md
  components/features/scalers/MinMaxNormalizer.md
  components/features/scalers/StandardNormalizer.md
  components/features/stationarity/FractionalDifference.md
  components/features/indicators/TAlibIndicator.md


.. toctree::
  :maxdepth: 1
  :caption: Actions

  components/actions/ActionStrategy.md
  components/actions/ContinuousActionStrategy.md
  components/actions/DiscreteActionStrategy.md
  components/actions/MultiDiscreteActionStrategy.md


.. toctree::
  :maxdepth: 1
  :caption: Rewards
  
  components/rewards/RewardStrategy.md
  components/rewards/RiskAdjustedReturnStrategy.md
  components/rewards/SimpleProfitStrategy.md


.. toctree::
  :maxdepth: 1
  :caption: Learning Agents

  agents/learning_agents.md
  agents/stable_baselines.md
  agents/tensorforce.md


.. toctree::
  :maxdepth: 1
  :caption: Strategies
  
  components/strategies/TradingStrategy.md
  components/strategies/StableBaselinesTradingStrategy.md
  components/strategies/TensorforceTradingStrategy.md


.. toctree::
   :maxdepth: 1
   :caption: API reference

   API reference <api/modules>
