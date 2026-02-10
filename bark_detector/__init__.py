"""
Bark Detector Pipeline
======================
Multi-layer filtering system to ensure audio samples contain dog barks.

Modules:
  - model.py           : CNN bark classifier (bark vs non-bark)
  - acoustic_filter.py  : Heuristic acoustic feature filtering
  - train_bark_detector.py : Training script using existing breed audio
  - youtube_scraper.py  : YouTube scraper with automatic bark filtering
  - validate.py         : Manual validation / review tool
  - pipeline.py         : Orchestrator tying all layers together
"""
