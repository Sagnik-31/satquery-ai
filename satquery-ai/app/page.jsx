"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import MapViewer from "../components/MapViewer";
import {
    ArrowDownToLine,
    ArrowRight,
    ArrowUp,
    Check,
    ChevronDown,
    ChevronRight,
    CircleHelp,
    ExternalLink,
    Eye,
    Globe2,
    History,
    ImagePlus,
    Layers3,
    LoaderCircle,
    MapPin,
    Maximize2,
    MessageSquare,
    Orbit,
    PanelLeftClose,
    Plus,
    Radar,
    ScanLine,
    Upload,
    X,
} from "lucide-react";
import { demoFeatures, download, getDemoResult, scenes, validateGeoJSON, validatePair, } from "../lib/satquery";
import { readImage } from "../lib/imagery";

const modes = [{id: "single", label: "Single image", icon: ScanLine}, {
    id: "temporal", label: "Before / after", icon: History
}, {id: "fusion", label: "SAR", icon: Radar},];
const empty = {type: "FeatureCollection", features: []};

export default function Home() {
    const [scene, setScene] = useState(scenes[0]);
    const [mode, setMode] = useState("single");
    const [base, setBase] = useState("satellite");
    const [visible, setVisible] = useState({
        water: true, urban: true, vegetation: true,
    });

    const [opacity, setOpacity] = useState(85);
    const [features, setFeatures] = useState(demoFeatures(scenes[0]));
    const [messages, setMessages] = useState([]);
    const [query, setQuery] = useState("");
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");
    const [selected, setSelected] = useState("");
    const [tab, setTab] = useState("assistant");
    const [focus, setFocus] = useState(0);

    const [modal, setModal] = useState(null);
    const [images, setImages] = useState([]);
    const [imageIndex, setImageIndex] = useState(0);
    const [benchmark, setBenchmark] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [sidebar, setSidebar] = useState(() => window.innerWidth > 960);
    const [drawing, setDrawing] = useState(false);
    const [bounds, setBounds] = useState(null);

    const fileRef = useRef(null);
    const geoRef = useRef(null);
    const scrollRef = useRef(null);
    const requestRef = useRef(null);
    const dialogRef = useRef(null);

    const last = messages.at(-1)?.result;
    let pairStatus = "";
    if (images.length === 2) {
        try {
            validatePair(images, mode);
            pairStatus = "CRS, dimensions, and extent match. Pixel alignment is not checked.";
        } catch (e) {
            pairStatus = e.message;
        }
    }

    const select = useCallback((name) => setSelected(name), []);
    const setArea = useCallback((b) => {
        setBounds(b);
    }, []);

    useEffect(() => {
        scrollRef.current?.scrollTo({
            top: scrollRef.current.scrollHeight, behavior: "smooth",
        });
    }, [messages, busy]);

    useEffect(() => () => {
        requestRef.current?.abort();
    }, [],);

    useEffect(() => {
        if (!modal) return;

        const previous = document.activeElement;
        const el = dialogRef.current;
        const first = el?.querySelector("button,input,select");

        first?.focus();
        const key = (e) => {
            if (e.key === "Escape") {
                setModal(null);
                return;
            }

            if (e.key === "Tab") {
                const all = el?.querySelectorAll("button:not(:disabled),input:not(:disabled),select,textarea,a[href]",);
                if (!all?.length) return;
                const a = all[0], b = all[all.length - 1];
                if (e.shiftKey && document.activeElement === a) {
                    e.preventDefault();
                    b.focus();
                } else if (!e.shiftKey && document.activeElement === b) {
                    e.preventDefault();
                    a.focus();
                }
            }
        };

        document.addEventListener("keydown", key);
        return () => {
            document.removeEventListener("keydown", key);
            previous?.focus();
        };
    }, [modal]);

    function reset(s = scene, m = mode) {
        requestRef.current?.abort();

        setBusy(false);
        setScene(s);
        setMode(m);
        setMessages([]);
        setFeatures(demoFeatures(s));
        setError("");
        setSelected("");
        setBounds(null);
    }

    async function run(text = query) {
        if (!text.trim() || busy) return;

        setError("");
        setBusy(true);
        setQuery("");

        const controller = new AbortController();
        requestRef.current = controller;

        try {
            let result;
            if (images.length) throw new Error("Remove uploaded images to use the scene queries.",);

            await new Promise((resolve, reject) => {
                const id = setTimeout(resolve, 500);
                controller.signal.addEventListener("abort", () => {
                    clearTimeout(id);
                    reject(new DOMException("Cancelled", "AbortError"));
                });
            });

            result = getDemoResult(text, mode, scene);
            if (controller.signal.aborted) return;
            setMessages((v) => [...v, {id: crypto.randomUUID(), query: text, result},]);
            setFeatures(result.features);
            setTab("assistant");
        } catch (e) {
            if (!controller.signal.aborted) {
                setError(e instanceof Error ? e.message : "Analysis failed.");
                setQuery(text);
            }
        } finally {
            if (requestRef.current === controller) setBusy(false);
        }
    }

    function exportResult() {
        if (!last) {
            setError("Run a query to create a report first.");
            return;
        }

        download("satquery-report.json", JSON.stringify({
            scene: scene.name, mode, createdAt: new Date().toISOString(), area: bounds, ...last,
        }, null, 2,),);
    }

    async function upload(list) {
        if (!list?.length) return;

        setUploading(true);
        setError("");

        const added = [];
        try {
            const limit = mode === "single" ? 1 : 2;
            if (images.length + list.length > limit) throw new Error(`This accepts ${limit} image${limit === 1 ? "" : "s"}. Remove an image or change workflow.`,);
            for (const file of Array.from(list)) added.push(await readImage(file, benchmark));

            setImages((v) => [...v, ...added]);
            setFeatures(empty);
            setMessages([]);
        } catch (e) {
            added.forEach((i) => {
                if (i.url.startsWith("blob:")) URL.revokeObjectURL(i.url);
            });
            setError((e).message);
        } finally {
            setUploading(false);
            if (fileRef.current) fileRef.current.value = "";
        }
    }

    const suggestions = mode === "single" ? ["Highlight the water bodies", "Describe the land cover", "Where are the built-up regions?",] : mode === "temporal" ? ["What changed between these dates?", "Has the built-up area increased?",] : ["Use optical and SAR to identify water", "Describe the combined scene",];

    const removeImage = id => {
        setImages((v) => {
            const next = v.filter((i) => i.id !== id);
            const old = v.find((i) => i.id === id);

            if (old?.url.startsWith("blob:")) URL.revokeObjectURL(old.url);
            if (!next.length) setFeatures(demoFeatures(scene));

            return next;
        });
        setImageIndex(0);
    };

    return (
        <main className="workspace">
            <header className="topbar">
                <a className="brand" href="/" aria-label="SatQuery home">
                  <span className="brand-symbol">
                    <Orbit size={25} />
                  </span>
                    <span>SatQuery<span className="brand-ai">AI</span>
          </span>
                </a>
                <span className="header-divider" />
                <div className="top-actions">
                    <button
                        className="mobile-layers quiet"
                        aria-label="Toggle explorer"
                        onClick={() => setSidebar(!sidebar)}
                    >
                        <Layers3 size={19} />
                    </button>
                    <button
                        className="quiet help"
                        title="Help"
                        aria-label="Help"
                        onClick={() => setModal("help")}
                    >
                        <CircleHelp size={18} />
                    </button>
                </div>
            </header>
            <nav className="activity-rail" aria-label="Workspace tools">
                <button
                    className={sidebar ? "rail-active" : ""}
                    title="Layers"
                    aria-label="Toggle layers"
                    onClick={() => setSidebar(!sidebar)}
                >
                    <Layers3 size={21} />
                </button>
                <button
                    title="Add imagery"
                    aria-label="Add imagery"
                    onClick={() => setModal("upload")}
                >
                    <ImagePlus size={21} />
                </button>
                <button
                    title="Query history"
                    aria-label="Query history"
                    onClick={() => setTab(tab === "history" ? "assistant" : "history")}
                >
                    <History size={21} />
                </button>
                <div className="rail-bottom">
                    <span>v1.0</span>
                </div>
            </nav>
            <div className="body-layout">
                {sidebar && (<aside className="left-panel">
                    <div className="panel-title">
                        <h2>Explorer</h2>
                        <button
                            aria-label="Collapse explorer"
                            className="quiet"
                            onClick={() => setSidebar(false)}
                        >
                            <PanelLeftClose size={17} />
                        </button>
                    </div>
                    <div className="scene-section">
                        <label className="eyebrow" htmlFor="scene">
                            AREA OF INTEREST
                        </label>
                        <div className="scene-select">
                            <MapPin size={16} />
                            <select
                                id="scene"
                                disabled={busy || images.length > 0}
                                value={scene.id}
                                onChange={(e) => reset(scenes.find((s) => s.id === e.target.value))}
                            >
                                {scenes.map((s) => (<option key={s.id} value={s.id}>
                                    {s.name}, India
                                </option>))}
                            </select>
                        </div>
                        <div className="scene-caption">
                            {scene.region}
                            <span>↗</span>
                        </div>
                    </div>
                    <div className="section-heading">
                        <span>IMAGERY</span>
                        <button
                            className="quiet"
                            aria-label="Upload imagery"
                            onClick={() => setModal("upload")}
                        >
                            <Plus size={15} />
                        </button>
                    </div>
                    <button className="imagery-card" onClick={() => setModal("upload")}>
                        <div className="image-thumb">
                            <Globe2 size={21} />
                            <span>RGB</span>
                        </div>
                        <div>
                            <strong>{images[0]?.name || "World imagery"}</strong>
                            <small>
                                {images.length ? "Uploaded imagery" : "Esri satellite basemap"}
                            </small>
                            <span className="tiny-tag">
                  {images.length ? "LOCAL PREVIEW" : "REFERENCE LAYER"}
                </span>
                        </div>
                        <Check size={15} />
                    </button>
                    <div className="source-note">
                        Basemap acquisition dates vary.
                    </div>
                    <button
                        className="upload-button"
                        onClick={() => setModal("upload")}
                    >
                        <Upload size={15} /> Add satellite imagery
                    </button>
                    <div className="section-heading layer-heading">
                        <span>MAP LAYERS</span>
                        <span className="count">
                {Object.values(visible).filter(Boolean).length}
              </span>
                    </div>
                    <div className="layer-list">
                        {([{
                            key: "water", label: "Water regions", subtitle: "Grounding overlay",
                        }, {
                            key: "urban", label: "Built-up regions", subtitle: "Detection overlay",
                        }, {
                            key: "vegetation", label: "Vegetation", subtitle: "Land-cover overlay",
                        },]).map((l) => (<label className="layer-row" key={l.key}>
                            <input
                                type="checkbox"
                                checked={visible[l.key]}
                                onChange={(e) => setVisible({...visible, [l.key]: e.target.checked})}
                            />
                            <span className={`layer-swatch ${l.key}`} />
                            <span>
                    <strong>{l.label}</strong>
                    <small>{l.subtitle}</small>
                  </span>
                            <Eye size={14} />
                        </label>))}
                    </div>
                    <div className="opacity-label">
                        <label htmlFor="opacity">Overlay opacity</label>
                        <span>{opacity}%</span>
                    </div>
                    <input
                        id="opacity"
                        className="opacity-slider"
                        type="range"
                        min="0"
                        max="100"
                        value={opacity}
                        onChange={(e) => setOpacity(+e.target.value)}
                    />
                    <button
                        className="import-button"
                        onClick={() => geoRef.current?.click()}
                    >
                        <Plus size={14} /> Import GeoJSON
                    </button>
                    <div className="basemap-section">
                        <div className="eyebrow">BASEMAP</div>
                        <div className="basemap-options">
                            <button
                                className={base === "satellite" ? "active" : ""}
                                onClick={() => setBase("satellite")}
                            >
                                <span className="sat-preview" />
                                <span>Satellite</span>
                                {base === "satellite" && <Check size={12} />}
                            </button>
                            <button
                                className={base === "streets" ? "active" : ""}
                                onClick={() => setBase("streets")}
                            >
                                <span className="street-preview" />
                                <span>Streets</span>
                                {base === "streets" && <Check size={12} />}
                            </button>
                        </div>
                    </div>
                </aside>)}
                <section className="map-section">
                    <div className="map-topline">
                        <div>
                            <span className="eyebrow">WORKSPACE /</span>
                            <h1>
                                {scene.name} <span>Region exploration</span>
                            </h1>
                        </div>
                        <button
                            title="Recenter map"
                            aria-label="Recenter map"
                            className="quiet"
                            onClick={() => setFocus((v) => v + 1)}
                        >
                            <Maximize2 size={17} />
                        </button>
                    </div>
                    <div className="map-stage">
                        <MapViewer
                            scene={scene}
                            features={features}
                            visible={visible}
                            opacity={opacity / 100}
                            base={base}
                            images={images}
                            imageIndex={imageIndex}
                            onSelect={select}
                            onBounds={setArea}
                            drawing={drawing}
                            setDrawing={setDrawing}
                            focus={focus}
                        />
                        <div className="map-context">
                            {images.length ? "Uploaded imagery" : base === "satellite" ? "Satellite view" : "Street map"}
                            <span className="context-divider" />{" "}
                            {images.length ? images[imageIndex]?.crs : "Reference basemap"}
                        </div>
                        {selected && (<div className="feature-card">
                            <button
                                className="quiet close"
                                aria-label="Close region details"
                                onClick={() => setSelected("")}
                            >
                                <X size={15} />
                            </button>
                            <span className="eyebrow">SELECTED REGION</span>
                            <strong>{selected}</strong>
                            <small>
                                Map region
                            </small>
                        </div>)}
                        {images[imageIndex] && !images[imageIndex].bounds && (<div className="unreferenced-preview">
                            <img
                                src={images[imageIndex].url}
                                alt={images[imageIndex].name}
                            />
                            <span>Image preview · no geographic coordinates</span>
                        </div>)}
                        <div className="map-legend">
              <span className="legend-title">
                MAP OVERLAYS
              </span>
                            {Object.entries(visible)
                                .filter(([, v]) => v)
                                .map(([key]) => (<span key={key}>
                    <i className={key} />
                                    {key === "urban" ? "Built-up" : key[0].toUpperCase() + key.slice(1)}
                  </span>))}
                        </div>
                        {bounds && (<button
                            className="area-chip"
                            onClick={() => {
                                setBounds(null);
                                setFocus((v) => v + 1);
                            }}
                        >
                            Selected area saved <X size={12} />
                        </button>)}
                    </div>
                    {mode !== "single" && (<div className="compare-strip">
                        <div>
                            <History size={16} />
                            <span>
                  {mode === "temporal" ? "Image comparison" : "Sensor comparison"}
                </span>
                            <small>
                                {images.length === 2 ? "Switch the visible image" : "Upload a paired dataset to compare"}
                            </small>
                        </div>
                        {[0, 1].map((i) => (<button
                            key={i}
                            disabled={!images[i]}
                            className={imageIndex === i ? "active" : ""}
                            onClick={() => setImageIndex(i)}
                        >
                            {mode === "temporal" ? i === 0 ? "Before" : "After" : i === 0 ? "Image 1" : "Image 2"}
                            {images[i]?.date && <small>{images[i].date}</small>}
                        </button>))}
                    </div>)}
                </section>
                <aside className="assistant-panel">
                    <div className="assistant-tabs">
                        <button
                            className={tab === "assistant" ? "active" : ""}
                            onClick={() => setTab("assistant")}
                        >
                            Assistant
                        </button>
                        <button
                            className={tab === "history" ? "active" : ""}
                            onClick={() => setTab("history")}
                        >
                            <History size={15} />
                            History{messages.length > 0 && <span>{messages.length}</span>}
                        </button>
                        <button
                            className="new-session"
                            title="New session"
                            aria-label="New session"
                            disabled={busy}
                            onClick={() => reset()}
                        >
                            <Plus size={18} />
                        </button>
                    </div>
                    <div className="assistant-scroll" ref={scrollRef}>
                        {tab === "history" ? (<div className="history-view">
                            <div className="eyebrow">THIS SESSION</div>
                            <h2>Query history</h2>
                            {messages.length === 0 ? (
                                <p>No queries yet.</p>) : (messages.map((m) => (<button
                                key={m.id}
                                onClick={() => {
                                    setFeatures(m.result.features);
                                    setTab("assistant");
                                    setSelected(m.result.title);
                                }}
                            >
                                <MessageSquare size={16} />
                                <span>
                        {m.query}
                                    <small>{m.result.task}</small>
                      </span>
                                <ChevronRight size={15} />
                            </button>)))}
                        </div>) : (<>
                            <div className="assistant-intro">
                                <h2>Scene analysis</h2>
                                <p>Select a workflow and ask about the scene.</p>
                            </div>
                            <div className="workflow-label">Analysis workflow</div>
                            <div className="mode-picker">
                                {modes.map((m) => (<button
                                    key={m.id}
                                    disabled={busy || images.length > 0}
                                    className={mode === m.id ? "active" : ""}
                                    onClick={() => reset(scene, m.id)}
                                >
                                    <m.icon size={15} />
                                    {m.label}
                                </button>))}
                            </div>
                            <div className="context-card">
                                <div className="context-icon">
                                    <ScanLine size={18} />
                                </div>
                                <div>
                                    <strong>
                                        {images.length ? images[0].name : scene.name}
                                    </strong>
                                    <small>{images.length ? "Uploaded imagery" : "Selected scene"}</small>
                                </div>
                            </div>
                            {messages.length === 0 && (<div className="suggestions">
                                <span className="eyebrow">EXAMPLE QUERIES</span>
                                {suggestions.map((s, i) => (<button key={s} onClick={() => run(s)} disabled={busy}>
                        <span>
                          {i === 0 ? (<ScanLine size={16} />) : i === 1 ? (<Layers3 size={16} />) : (
                              <MapPin size={16} />)}
                        </span>
                                    {s}
                                    <ArrowRight size={14} />
                                </button>))}
                            </div>)}
                            {messages.map((m) => (<div className="exchange" key={m.id}>
                                <div className="user-query">{m.query}</div>
                                <div className="result-header">
                                    <strong>SatQuery</strong>
                                </div>
                                <h3>{m.result.title}</h3>
                                <p className="answer">{m.result.answer}</p>
                                <div className="result-metrics">
                                    {m.result.metrics.map((x, i) => (<div key={i}>
                                        <strong>{x.value}</strong>
                                        <span>{x.label}</span>
                                    </div>))}
                                </div>
                                <button
                                    className="view-evidence"
                                    onClick={() => {
                                        setFeatures(m.result.features);
                                        setVisible({
                                            water: true, urban: true, vegetation: true,
                                        });
                                        setFocus((v) => v + 1);
                                    }}
                                >
                                    {" "}
                                    <Layers3 size={14} /> Show spatial evidence{" "}
                                    <ArrowRight size={14} />
                                </button>
                            </div>))}
                            {busy && (<div className="thinking" role="status">
                                <LoaderCircle size={15} className="spin" />
                                Preparing results…
                                <button
                                    onClick={() => {
                                        requestRef.current?.abort();
                                        setBusy(false);
                                    }}
                                >
                                    Cancel
                                </button>
                            </div>)}
                        </>)}
                    </div>
                    <div className="composer-area">
                        {error && (<div className="error-message" role="alert">
                            {error}
                            <button aria-label="Dismiss error" onClick={() => setError("")}>
                                <X size={13} />
                            </button>
                        </div>)}
                        <form
                            className="composer"
                            onSubmit={(e) => {
                                e.preventDefault();
                                run();
                            }}
                        >
              <textarea
                  aria-label="Ask about the scene"
                  placeholder="Ask a question about the scene…"
                  value={query}
                  maxLength={2000}
                  rows={2}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          run();
                      }
                  }}
              />
                            <div>
                                <button
                                    type="button"
                                    title="Attach images"
                                    aria-label="Attach images"
                                    onClick={() => setModal("upload")}
                                >
                                    <Plus size={18} />
                                </button>
                                <span>
                  {mode === "single" ? "Single image" : mode === "temporal" ? "Before / after" : "Optical + SAR"}
                                    <ChevronDown size={11} />
                </span>
                                <button
                                    className="send-button"
                                    type="submit"
                                    disabled={busy || !query.trim()}
                                    aria-label="Send query"
                                >
                                    <ArrowUp size={17} />
                                </button>
                            </div>
                        </form>
                    </div>
                    <div className="export-bar">
                        <button disabled={!last} onClick={exportResult}>
                            <ArrowDownToLine size={15} />
                            Export report
                        </button>
                        <button
                            disabled={!features.features.length}
                            onClick={() => {
                                download("satquery-regions.geojson", JSON.stringify(features, null, 2), "application/geo+json",);
                            }}
                        >
                            GeoJSON <ExternalLink size={12} />
                        </button>
                    </div>
                </aside>
            </div>
            <input
                type="file"
                ref={geoRef}
                accept=".geojson,.json"
                hidden
                onChange={async (e) => {
                    const f = e.target.files?.[0];
                    if (!f) return;
                    try {
                        if (f.size > 5 * 1024 * 1024) throw new Error("Use a GeoJSON file smaller than 5 MB.");
                        setFeatures(validateGeoJSON(JSON.parse(await f.text())));
                        setSelected("");
                    } catch (err) {
                        setError((err).message);
                    }
                    e.target.value = "";
                }}
            />
            {modal && (<div
                className="modal-backdrop"
                onClick={(e) => {
                    if (e.target === e.currentTarget) setModal(null);
                }}
            >
                <div
                    ref={dialogRef}
                    className="modal"
                    role="dialog"
                    aria-modal="true"
                    aria-labelledby="dialog-title"
                >
                    <button
                        className="quiet modal-close"
                        aria-label="Close dialog"
                        onClick={() => setModal(null)}
                    >
                        <X size={20} />
                    </button>
                    {modal === "upload" ? (<>
                <span className="modal-icon">
                  <ImagePlus size={23} />
                </span>
                        <h2 id="dialog-title">Add satellite images</h2>
                        <p>
                            Add{" "}
                            {mode === "single" ? "one image" : "two co-registered images"}{" "}
                            for viewing.
                        </p>
                        <div
                            className="dropzone"
                            onDragOver={(e) => e.preventDefault()}
                            onDrop={(e) => {
                                e.preventDefault();
                                if (!uploading) upload(e.dataTransfer.files);
                            }}
                        >
                            <Upload size={27} />
                            <strong>
                                {uploading ? "Reading imagery…" : "Drop satellite imagery here"}
                            </strong>
                            <span>GeoTIFF / TIFF · up to 30 MB per image</span>
                            <button
                                disabled={uploading}
                                onClick={() => fileRef.current?.click()}
                            >
                                Choose files
                            </button>
                            <input
                                ref={fileRef}
                                hidden
                                type="file"
                                multiple={mode !== "single"}
                                accept=".tif,.tiff,.png,.jpg,.jpeg"
                                onChange={(e) => upload(e.target.files)}
                            />
                        </div>
                        <label className="benchmark-check">
                            <input
                                type="checkbox"
                                checked={benchmark}
                                onChange={(e) => setBenchmark(e.target.checked)}
                            />{" "}
                            These PNG/JPEG files come from an approved benchmark dataset
                        </label>
                        {images.map((img) => (<div className="uploaded-image" key={img.id}>
                            <img src={img.url} alt="Uploaded preview" />
                            <div>
                                <strong>{img.name}</strong>
                                <small>
                                    {img.width} × {img.height} · {img.crs}
                                </small>
                                <div className="image-meta">
                                    <select
                                        aria-label={`Modality for ${img.name}`}
                                        value={img.modality}
                                        onChange={(e) => setImages((v) => v.map((x) => x.id === img.id ? {
                                            ...x, modality: e.target.value,
                                        } : x,),)}
                                    >
                                        <option value="optical">Optical</option>
                                        <option value="sar">SAR</option>
                                    </select>
                                    <input
                                        aria-label={`Acquisition date for ${img.name}`}
                                        type="date"
                                        value={img.date}
                                        onChange={(e) => setImages((v) => v.map((x) => x.id === img.id ? {
                                            ...x, date: e.target.value
                                        } : x,),)}
                                    />
                                </div>
                            </div>
                            <button
                                className="quiet"
                                aria-label={`Remove ${img.name}`}
                                onClick={() => removeImage(img.id)}
                            >
                                <X size={15} />
                            </button>
                        </div>))}
                        {pairStatus && <p className="pair-check">{pairStatus}</p>}
                        {error && (<p role="alert" className="dialog-error">
                            {error}
                        </p>)}
                        <div className="modal-footer">
                  <span>
                    Images are processed locally in this browser.
                  </span>
                            <button
                                className="primary"
                                disabled={uploading}
                                onClick={() => setModal(null)}
                            >
                                Open workspace <ArrowRight size={14} />
                            </button>
                        </div>
                    </>) : (<>
                <span className="modal-icon">
                  <Orbit size={25} />
                </span>
                        <h2 id="dialog-title">Using the workspace</h2>
                        <p>
                            Select a workflow, ask a suggested question, then inspect the
                            highlighted regions on the map.
                        </p>
                        <ol className="help-steps">
                            <li>
                                <strong>Single image</strong>Ask a scene question or
                                highlight a water region.
                            </li>
                            <li>
                                <strong>Before / after</strong>Show example change regions,
                                or upload a dated image pair.
                            </li>
                            <li>
                                <strong>Optical + SAR</strong>View a paired-sensor
                                analysis workflow.
                            </li>
                            <li>
                                <strong>Inspect & export</strong>Toggle layers, select a
                                region, and download the report or GeoJSON.
                            </li>
                        </ol>
                        <button className="primary" onClick={() => setModal(null)}>
                            Close guide <ArrowRight size={14} />
                        </button>
                    </>)}
                </div>
            </div>)}
        </main>);
}
