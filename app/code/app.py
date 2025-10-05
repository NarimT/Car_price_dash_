# app.py — minimal, uses self.W if present (handles your custom LogisticRegression)
from __future__ import annotations
import traceback
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# mlflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow_available = True
except Exception:
    mlflow_available = False
    mlflow = None
    MlflowClient = None

# ---------- CONFIG ----------
TRACKING_URI = "https://admin:password@mlflow.ml.brain.cs.ait.ac.th"
MODEL_NAME = "A3_st125983"
MODEL_VERSION = "1"

# simple encoders
BRAND_LIST = [
    'Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Honda', 'Ford', 'Toyota',
    'Chevrolet', 'Renault', 'Volkswagen', 'Nissan', 'Skoda', 'BMW', 'Mercedes-Benz'
]
BRAND_MAP = {b: i for i, b in enumerate(BRAND_LIST)}
FUEL_MAP = {'Petrol': 0, 'Diesel': 1, 'Electric': 2}

# global
_model = None
_model_info = "Model not loaded."
_model_trace = None

def load_model():
    global _model, _model_info, _model_trace
    if not mlflow_available:
        _model = None
        _model_info = "mlflow not installed"
        _model_trace = None
        return
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        _model = mlflow.pyfunc.load_model(uri)
        _model_info = f"✅ Loaded MLflow model: {MODEL_NAME} (version {MODEL_VERSION})"
        _model_trace = None
    except Exception as e:
        _model = None
        _model_info = f"❌ Failed to load model: {e}"
        _model_trace = traceback.format_exc()

load_model()

def encode_brand(b):
    return float(BRAND_MAP.get(str(b), abs(hash(str(b))) % 1000))

def encode_fuel(f):
    return float(FUEL_MAP.get(str(f), 99))

def build_X_with_intercept(year, power, mileage, brand, fuel):
    # Order: [intercept, year, max_power, mileage, brand_idx, fuel_idx]
    i = 1.0
    y = float(year)
    p = float(str(power).replace(",", "."))
    m = float(str(mileage).replace(",", "."))
    b = encode_brand(brand)
    f = encode_fuel(fuel)
    X = np.array([[i, y, p, m, b, f]], dtype=np.float64)
    return X

def find_weights(obj):
    """
    Try to find weights in the provided object.
    Returns tuple (weights_array_or_None, attr_name_or_None).
    Accepts:
      - coef_ (+ intercept_)
      - W, W_, weights, theta, Theta, etc.
    """
    candidates = [
        ("coef_", "coef_"),
        ("W", "W"),
        ("W_", "W_"),
        ("weights", "weights"),
        ("theta", "theta"),
        ("Theta", "Theta"),
        ("_W", "_W")
    ]
    # first check top-level attributes
    for attr_key, attr_name in candidates:
        w = getattr(obj, attr_key, None)
        if w is not None:
            return w, attr_key

    # If pyfunc wrapper, common place: obj._model_impl.sklearn_model or similar
    impl = getattr(obj, "_model_impl", None) or getattr(obj, "_PyFuncModel__model_impl", None)
    if impl is not None:
        # often sklearn_model attribute holds the original estimator
        sk = getattr(impl, "sklearn_model", None) or getattr(impl, "model", None)
        if sk is not None:
            for attr_key, attr_name in candidates:
                w = getattr(sk, attr_key, None)
                if w is not None:
                    return w, f"{'impl.'}{attr_key}"
            # also try nested attributes in common wrappers (Pipeline, final_estimator)
            nested_names = ["final_estimator", "estimator_", "named_steps", "steps"]
            for nn in nested_names:
                sub = getattr(sk, nn, None)
                if sub is None:
                    continue
                # if dict-like named_steps
                if hasattr(sub, "items"):
                    for k, v in list(sub.items()):
                        for attr_key, _ in candidates:
                            w = getattr(v, attr_key, None)
                            if w is not None:
                                return w, f"impl.{nn}.{k}.{attr_key}"
                elif isinstance(sub, (list, tuple)):
                    for i, v in enumerate(sub):
                        comp = getattr(v, 1, v) if isinstance(v, tuple) and len(v) > 1 else v
                        for attr_key, _ in candidates:
                            w = getattr(comp, attr_key, None)
                            if w is not None:
                                return w, f"impl.{nn}[{i}].{attr_key}"
    return None, None

# Dash UI
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H3("Minimal manual prediction using W (if present)"),
    html.Div(_model_info, id="model_info", style={"fontWeight":"bold"}),
    html.Pre(id="model_trace", style={"whiteSpace":"pre-wrap", "color":"darkred"}),

    html.Label("Year"), dcc.Input(id="year", type="number", value=2017), html.Br(),
    html.Label("Max power (hp)"), dcc.Input(id="power", type="text", value="82.4"), html.Br(),
    html.Label("Mileage"), dcc.Input(id="mileage", type="text", value="21.14"), html.Br(),
    html.Label("Brand"), dcc.Input(id="brand", type="text", value="Maruti"), html.Br(),
    html.Label("Fuel"), dcc.Dropdown(id="fuel", options=[
        {"label":"Petrol","value":"Petrol"},
        {"label":"Diesel","value":"Diesel"},
        {"label":"Electric","value":"Electric"}], value="Petrol"), html.Br(),

    html.Button("Predict (manual)", id="btn", n_clicks=0),
    html.Hr(),
    html.Div(id="out_pred", style={"fontWeight":"600"}),
    html.Pre(id="out_debug", style={"whiteSpace":"pre-wrap", "color":"gray"})
])

@app.callback(
    Output("out_pred", "children"),
    Output("out_debug", "children"),
    Output("model_info", "children"),
    Output("model_trace", "children"),
    Input("btn", "n_clicks"),
    State("year", "value"),
    State("power", "value"),
    State("mileage", "value"),
    State("brand", "value"),
    State("fuel", "value"),
)
def on_click(n_clicks, year, power, mileage, brand, fuel):
    info = _model_info
    trace = _model_trace or ""
    if n_clicks == 0:
        return "", "Ready.", info, trace

    # parse inputs
    try:
        y = int(year) if year is not None else 0
        p = float(str(power).replace(",", ".")) if power is not None else 0.0
        m = float(str(mileage).replace(",", ".")) if mileage is not None else 0.0
        b = str(brand) if brand is not None else ""
        f = str(fuel) if fuel is not None else ""
    except Exception as e:
        return "Input parse error", traceback.format_exc(), info, trace

    X = build_X_with_intercept(y, p, m, b, f)  # shape (1,6)
    dbg = []
    dbg.append(f"Built X (intercept first):\n{X}")
    dbg.append(f"X.shape = {X.shape} dtype={X.dtype}")

    if _model is None:
        dbg.append("Model not loaded; cannot inspect weights.")
        return "Model not loaded", "\n".join(dbg), info, trace

    # try to find weights
    W, source = find_weights(_model)
    if W is None:
        dbg.append("No weights found (tried coef_, W, W_, weights, theta, Theta, and searching impl.sklearn_model).")
        dbg.append("You have a custom estimator. It stores trained weights in `self.W`. The code searched for that and equivalents but didn't find them.")
        dbg.append("Found top-level model type: " + str(type(_model)))
        # show some hints: print impl.sklearn_model type if present
        impl = getattr(_model, "_model_impl", None) or getattr(_model, "_PyFuncModel__model_impl", None)
        if impl is not None:
            sk = getattr(impl, "sklearn_model", None) or getattr(impl, "model", None)
            dbg.append("impl type: " + str(type(impl)))
            dbg.append("impl.sklearn_model type: " + str(type(sk)))
            # try to print available attr names on sk
            try:
                dbg.append("Attributes on impl.sklearn_model (first 80 chars): " + ", ".join(dir(sk)[:40]))
            except Exception:
                pass
        return "Weights not found", "\n".join(dbg), info, trace

    # numpy-ify W
    try:
        W_np = np.asarray(W, dtype=np.float64)
    except Exception:
        # some custom structures may require conversion via .copy() or attribute access
        try:
            W_np = np.asarray(getattr(W, "W", W), dtype=np.float64)
        except Exception:
            return "Failed to cast found weights to numpy", traceback.format_exc(), info, trace

    dbg.append(f"Found weights from attribute: {source}")
    dbg.append(f"W.shape = {getattr(W_np, 'shape', str(type(W_np)))} dtype={getattr(W_np, 'dtype', 'unknown')}")

    # compute prediction: class does X @ W  (your class expects X shape (m,n), W shape (n,k))
    try:
        # ensure X dims
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # If W_np shape is (n, k) and X shape is (1, n) -> X @ W_np -> (1,k) OK
        # If W_np shape is (k, n) maybe transposed — try to detect and transpose if necessary
        if W_np.ndim == 1:
            # single-dim weights (n,) treat as coef for single output
            if W_np.shape[0] == X.shape[1]:
                linear = X.dot(W_np.reshape(-1,1)).flatten()  # single output
            else:
                dbg.append("1D W shape incompatible with X; W.shape=" + str(W_np.shape))
                return "Dimension mismatch", "\n".join(dbg), info, trace
        elif W_np.ndim == 2:
            n_w_row, n_w_col = W_np.shape
            if n_w_row == X.shape[1]:
                linear = X.dot(W_np)  # (1,k)
            elif n_w_col == X.shape[1]:
                # maybe stored transposed; use transpose
                linear = X.dot(W_np.T)
                dbg.append("Detected W stored transposed; used W.T for computation.")
            else:
                dbg.append(f"Dimension mismatch between X.shape {X.shape} and W.shape {W_np.shape}")
                return "Dimension mismatch", "\n".join(dbg), info, trace
        else:
            dbg.append(f"Unsupported W ndim: {W_np.ndim}")
            return "Unsupported W ndim", "\n".join(dbg), info, trace

        dbg.append(f"Computed raw linear scores (X @ W): {repr(linear)}")
        # interpret
        arr = np.array(linear)
        if arr.ndim == 2 and arr.shape[1] > 1:
            cls = int(np.argmax(arr, axis=1)[0])
            dbg.append(f"Predicted class (argmax): {cls}")
            return f"Manual prediction class: {cls}", "\n".join(dbg), info, trace
        else:
            # single output: round to nearest int as before
            val = float(np.ravel(arr)[0])
            cls = int(round(val))
            dbg.append(f"Predicted scalar (rounded): {val} -> {cls}")
            return f"Manual prediction scalar->class: {cls}", "\n".join(dbg), info, trace
    except Exception:
        return "Computation error", traceback.format_exc(), info, trace

if __name__ == "__main__":
    try:
        app.run(debug=True, host="0.0.0.0", port=8050)
    except AttributeError:
        app.run_server(debug=True, host="0.0.0.0", port=8050)
