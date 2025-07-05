# Data Dictionary

* **record_id** (`int64`): 
* **date** (`object`): Date of record (YYYY‑MM‑DD).
* **property_type** (`object`): Hotel or Flight.
* **location** (`object`): IATA airport code or city.
* **competitor_price** (`float64`): Price in USD.
* **dynamic_price** (`float64`): Price in USD.
* **demand_forecast** (`float64`): Modelled demand level (0‑1).
* **available_units** (`int64`): Number of units (rooms/seats).
* **units_sold** (`int64`): Number of units (rooms/seats).
* **revenue** (`float64`): 
* **resource_utilization** (`float64`): Ratio of sold / available.
* **customer_id** (`int64`): 
* **customer_age** (`int64`): 
* **customer_gender** (`object`): 
* **loyalty_tier** (`object`): 
* **booking_channel** (`object`): 
* **booking_lead_time_days** (`int64`): 
* **segment** (`object`): 
* **campaign_id** (`object`): 
* **promo_sent** (`bool`): 
* **promo_response** (`bool`): 
* **promo_click_rate** (`float64`): 
* **sentiment_score** (`float64`): Customer sentiment (-1 to 1).
* **ancillary_purchase** (`bool`): 
* **avg_spend_on_site** (`float64`): 
* **staff_hours** (`float64`): 
* **maintenance_flag** (`bool`): 
* **energy_consumption** (`float64`): 
* **delay_minutes** (`float64`): 
* **risk_level** (`object`): Operational risk bucket.
* **route_distance_km** (`float64`): 
* **flight_speed_knots** (`float64`): 
* **fdm_anomaly_flag** (`object`): True if flight data monitoring flagged anomaly.