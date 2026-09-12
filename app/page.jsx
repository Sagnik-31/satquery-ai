"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import MapViewer from "../components/MapViewer";
import {
    ArrowDownToLine,
    ArrowRight,
    ArrowUp,
    Check,
    ChevronDown,
    Download,
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
import { boundsToStacBbox, demoFeatures, download, getDemoResult, scenes, validateGeoJSON, validatePair, } from "../lib/satquery";
import { readImage } from "../lib/imagery";

const modes = [
    {id: "catalog", label: "Location + date", icon: Globe2},
    {id: "temporal", label: "Manual Before / After", icon: History},
    {id: "single", label: "Single image", icon: ScanLine},
    {id: "fusion", label: "SAR", icon: Radar},
];
const empty = {type: "FeatureCollection", features: []};
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function backendImageSrc(value) {
    return value ? `data:image/png;base64,${value}` : "";
}

function backendResultToFrontend(result) {
    const isCatalog = result.analysis_source === "satellite_catalog";
    return {
        kind: "change",
        category: result.category || "general",
        title: isCatalog
            ? (result.category === "general"
                ? "Satellite temporal change analysis"
                : `${result.category[0].toUpperCase()}${result.category.slice(1)} change from satellite imagery`)
            : (result.category === "general"
                ? "Bi-temporal change analysis"
                : `${result.category[0].toUpperCase()}${result.category.slice(1)} change intelligence`),
        answer: result.answer,
        task: isCatalog ? "Satellite catalog change detection" : "Semantic change detection",
        metrics: result.metrics || [],
        features: empty,
        warnings: result.warnings || [],
        visuals: result.visuals || {},
        semantic: result.semantic || null,
        disclaimer: result.disclaimer || "",
        alignment: result.alignment || {},
        technical: result.technical || {},
        method: result.method || [],
        processing: result.timing_ms ? `${(result.timing_ms / 1000).toFixed(2)}s` : "",
        analysisSource: result.analysis_source || "manual_upload",
        pairId: result.pair_id || null,
        location: result.location || null,
        before: result.before || null,
        after: result.after || null,
        provenance: result.provenance || null,
        generalChange: result.general_change || null,
    };
}


function DataProvenance({ result }) {
    if (!result?.provenance && !result?.before) return null;

    const prov = result.provenance || {};
    const before = result.before || prov.before || {};
    const after = result.after || prov.after || {};
    const loc = result.location || prov.location || {};

    return (
        <div className="provenance-panel">
            <h3>Data provenance</h3>
            <div className="provenance-grid">
                <div>
                    <span>Location</span>
                    <strong>{loc.name || "Selected AOI"}</strong>
                    {loc.latitude != null && (
                        <small>{loc.latitude.toFixed?.(4) ?? loc.latitude}, {loc.longitude.toFixed?.(4) ?? loc.longitude}</small>
                    )}
                </div>
                <div>
                    <span>AOI</span>
                    <strong>{prov.aoi_bbox_wgs84 ? "Custom bounding box" : `${loc.aoi_radius_km ?? 2.5} km radius`}</strong>
                    {prov.aoi_bbox_wgs84 && (
                        <small>{prov.aoi_bbox_wgs84.map((v) => v.toFixed(3)).join(", ")}</small>
                    )}
                </div>
                <div>
                    <span>T1 — Historical</span>
                    <strong>{before.sensor || "Sentinel-2"} · {before.date || "—"}</strong>
                    <small>{before.datetime_utc || ""}{before.cloud_cover != null ? ` · Cloud ${before.cloud_cover}%` : ""}</small>
                </div>
                <div>
                    <span>T2 — Latest available</span>
                    <strong>{after.sensor || "Sentinel-2"} · {after.date || "—"}</strong>
                    <small>{after.datetime_utc || ""}{after.cloud_cover != null ? ` · Cloud ${after.cloud_cover}%` : ""}</small>
                </div>
                <div>
                    <span>Source</span>
                    <strong>{before.source || prov.source_catalog || "Element84 Earth Search"}</strong>
                </div>
                <div>
                    <span>Spectral bands</span>
                    <strong>{prov.has_nir ? "RGB + NIR (NDVI/NDWI capable)" : "RGB only"}</strong>
                </div>
            </div>
            {(prov.processing || []).length > 0 && (
                <ul className="provenance-processing">
                    {prov.processing.map((step) => (
                        <li key={step}>{step}</li>
                    ))}
                </ul>
            )}
        </div>
    );
}


function TechnicalDetails({ result }) {
    const [open, setOpen] = useState(false);
    if (!result?.technical && !result?.alignment) return null;

    const tech = result.technical || {};
    const alignment = result.alignment || {};

    return (
        <details className="technical-details" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
            <summary>Advanced analysis</summary>
            <div className="technical-details-grid">
                <div>
                    <span>Alignment</span>
                    <strong>{alignment.success ? "Successful" : "Fallback used"}</strong>
                </div>
                <div>
                    <span>Threshold</span>
                    <strong>{tech.threshold ?? "—"}</strong>
                </div>
                <div>
                    <span>Min region area</span>
                    <strong>{tech.min_area_px ? `${tech.min_area_px}px` : "—"}</strong>
                </div>
                <div>
                    <span>Feature inliers</span>
                    <strong>{alignment.matches_or_inliers ?? "—"}</strong>
                </div>
                <div>
                    <span>Inlier ratio</span>
                    <strong>{alignment.inlier_ratio ?? "—"}</strong>
                </div>
                <div>
                    <span>Processing time</span>
                    <strong>{result.processing || "—"}</strong>
                </div>
                <div>
                    <span>Semantic backend</span>
                    <strong>{tech.semantic_backend || "none"}</strong>
                </div>
            </div>
            {(result.method || []).length > 0 && (
                <ul className="technical-method-list">
                    {result.method.map((step) => (
                        <li key={step}>{step}</li>
                    ))}
                </ul>
            )}
        </details>
    );
}

function TemporalComparison({ pair, result, lastQuery, busy }) {
    const before = pair?.before;
    const after = pair?.after;
    const visuals = result?.visuals || {};

    if (!before?.url || !after?.url) return null;

    const isSemantic = result?.category && result.category !== "general";
    const fallbackSemantic = result?.semantic?.backend === "deterministic_visual_fallback";
    const isCatalog = result?.analysisSource === "satellite_catalog";

    return (
        <div className="temporal-dashboard">
            {isCatalog && result?.location && (
                <div className="temporal-location-banner">
                    <h3>{result.location.name || "Selected location"}</h3>
                    <div className="temporal-date-row">
                        <span>T1: <strong>{result.before?.date || before.date || "—"}</strong></span>
                        <span>T2: <strong>{result.after?.date || after.date || "—"}</strong> (latest available acquisition)</span>
                    </div>
                </div>
            )}
            <div className="temporal-header">
                <div>
                    <span className="eyebrow">TEMPORAL CHANGE ANALYSIS</span>
                    <h2>{isCatalog ? "Historical vs Latest Available" : "Before / After"}</h2>
                    <p>
                        {isCatalog
                            ? "Sentinel-2 imagery retrieved for the same AOI, aligned, and analyzed for temporal change."
                            : "Side-by-side T1 and T2 imagery with general change detection and optional semantic category analysis."}
                    </p>
                </div>
                {busy ? (
                    <div className="temporal-status temporal-status-busy">
                        <LoaderCircle size={14} className="spin" /> Analyzing…
                    </div>
                ) : result ? (
                    <div className="temporal-status">
                        <span className="status-dot" /> Analysis ready
                    </div>
                ) : null}
            </div>

            <div className="temporal-pair-grid">
                <figure className="temporal-card">
                    <div className="temporal-card-head">
                        <div>
                            <span className="temporal-label">BEFORE / T1</span>
                            <strong>{before.name}</strong>
                            {before.subtitle && <span className="temporal-card-meta">{before.subtitle}</span>}
                            {before.cloud != null && <span className="temporal-card-meta">Cloud {before.cloud}%</span>}
                        </div>
                        {before.date && <span className="temporal-date">{before.date}</span>}
                    </div>
                    <img src={before.url} alt={`Before: ${before.name}`} />
                </figure>

                <figure className="temporal-card">
                    <div className="temporal-card-head">
                        <div>
                            <span className="temporal-label">{isCatalog ? "LATEST / T2" : "AFTER / T2"}</span>
                            <strong>{after.name}</strong>
                            {after.subtitle && <span className="temporal-card-meta">{after.subtitle}</span>}
                            {after.cloud != null && <span className="temporal-card-meta">Cloud {after.cloud}%</span>}
                        </div>
                        {after.date && <span className="temporal-date">{after.date}</span>}
                    </div>
                    <img src={after.url} alt={`After: ${after.name}`} />
                </figure>
            </div>

            {result && result.kind === "change" && (
                <>
                    <div className="temporal-section-label">
                        <span className="eyebrow">CHANGE MAP</span>
                    </div>
                    <div className="temporal-result-block">
                        <div className="temporal-result-head">
                            <div>
                                <h3>Detected visual change</h3>
                                <p className="temporal-summary">{result.answer}</p>
                            </div>
                            <div className="temporal-mini-metrics">
                                {(result.metrics || []).slice(0, 4).map((item, i) => (
                                    <div key={i}>
                                        <strong>{item.value}</strong>
                                        <span>{item.label}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="temporal-visual-grid">
                            {visuals.overlay_png && (
                                <figure className="temporal-visual-card temporal-visual-wide">
                                    <img src={backendImageSrc(visuals.overlay_png)} alt="Detected change regions" />
                                    <figcaption>Highlighted temporal change visualization</figcaption>
                                </figure>
                            )}
                            {visuals.heatmap_png && (
                                <figure className="temporal-visual-card">
                                    <img src={backendImageSrc(visuals.heatmap_png)} alt="Temporal difference heatmap" />
                                    <figcaption>Temporal difference heatmap</figcaption>
                                </figure>
                            )}
                        </div>

                        {result.warnings?.length > 0 && (
                            <div className="temporal-warnings">
                                {result.warnings.map((warning, i) => (
                                    <div className="result-warning" key={i}>{warning}</div>
                                ))}
                            </div>
                        )}
                    </div>

                    {isSemantic && (
                        <div className="temporal-semantic-section">
                            <div className="temporal-section-label">
                                <span className="eyebrow">SEMANTIC ANALYSIS</span>
                            </div>
                            {lastQuery && (
                                <div className="semantic-query-card">
                                    <span>Question</span>
                                    <p>{lastQuery}</p>
                                </div>
                            )}

                            {result.semantic ? (
                                <>
                                    <div className="semantic-answer-card">
                                        <div className="semantic-answer-head">
                                            <div>
                                                <span className="eyebrow">{result.semantic.category?.toUpperCase()} CHANGE</span>
                                                <h3>{result.semantic.category[0].toUpperCase() + result.semantic.category.slice(1)} analysis</h3>
                                            </div>
                                            <div className="semantic-confidence">
                                                <span>Semantic signal confidence</span>
                                                <strong>{((result.semantic.confidence || 0) * 100).toFixed(1)}%</strong>
                                            </div>
                                        </div>
                                        <p>{result.answer}</p>
                                        <div className="semantic-answer-metrics">
                                            <div><span>New {result.semantic.category}</span><strong>{Number(result.semantic.new_percentage || 0).toFixed(2)}%</strong></div>
                                            <div><span>{result.semantic.category} loss</span><strong>{Number(result.semantic.removed_percentage || 0).toFixed(2)}%</strong></div>
                                            <div><span>Direction</span><strong>{result.semantic.direction || "mixed"}</strong></div>
                                        </div>
                                        {fallbackSemantic && (
                                            <p className="semantic-fallback-note">
                                                Deterministic visual fallback active — suitable for demo, not geospatial ground truth.
                                            </p>
                                        )}
                                    </div>

                                    {visuals.semantic_overlay_png && (
                                        <figure className="temporal-visual-card temporal-visual-wide semantic-map-card">
                                            <img src={backendImageSrc(visuals.semantic_overlay_png)} alt="Semantic category change map" />
                                            <figcaption>
                                                {result.semantic.category[0].toUpperCase() + result.semantic.category.slice(1)} change map
                                            </figcaption>
                                        </figure>
                                    )}
                                </>
                            ) : (
                                <p className="temporal-summary">Semantic category analysis was unavailable for this query.</p>
                            )}
                        </div>
                    )}

                    <DataProvenance result={result} />
                    <TechnicalDetails result={result} />
                </>
            )}
        </div>
    );
}

export default function Home() {
    const [scene, setScene] = useState(scenes[0]);
    const [mode, setMode] = useState("catalog");
    const [catalogLocation, setCatalogLocation] = useState(`${scenes[0].name}, India`);
    const [beforeDate, setBeforeDate] = useState("2024-01-15");
    const [cloudThreshold, setCloudThreshold] = useState(20);
    const [catalogPairId, setCatalogPairId] = useState("");
    const [base, setBase] = useState("satellite");
    const [visible, setVisible] = useState({
        water: true, urban: true, vegetation: true,
    });

    const [opacity, setOpacity] = useState(85);
    const [minArea, setMinArea] = useState(3000);
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
    const [uploading, setUploading] = useState(false);
    const [autoPairKey, setAutoPairKey] = useState("");
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
            pairStatus = images[0]?.bounds && images[1]?.bounds
                ? "Georeferenced pair detected. Backend registration will verify spatial alignment."
                : "Image-only pair detected. Backend will align the pixels and compare the scene; GIS coordinates are not available for these uploads.";
            if (
                mode === "temporal" &&
                (images[0]?.width !== images[1]?.width || images[0]?.height !== images[1]?.height)
            ) {
                pairStatus += " Different image dimensions will be resized automatically by the backend.";
            }
        } catch (e) {
            pairStatus = e.message;
        }
    }

    const select = useCallback((name) => setSelected(name), []);
    const setArea = useCallback((b) => {
        setBounds(b);
        setCatalogPairId("");
    }, []);

    useEffect(() => {
        scrollRef.current?.scrollTo({
            top: scrollRef.current.scrollHeight, behavior: "smooth",
        });
    }, [messages, busy]);

    useEffect(() => () => {
        requestRef.current?.abort();
    }, [],);

    // As soon as a valid temporal pair is loaded, generate a general change
    // map automatically so the workspace is immediately useful.
    useEffect(() => {
        if (mode !== "temporal" || images.length !== 2 || busy) return;

        const key = [
            images[0]?.name,
            images[0]?.file?.size,
            images[1]?.name,
            images[1]?.file?.size,
        ].join("::");

        if (!key || key === autoPairKey) return;

        setAutoPairKey(key);
        run("What changed between these images?");
    }, [mode, images, busy, autoPairKey]);

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
        setAutoPairKey("");
        setCatalogPairId("");
    }

    function handleSceneChange(nextScene) {
        reset(nextScene, mode);
        setCatalogLocation(`${nextScene.name}, India`);
        setCatalogPairId("");
    }

    function switchMode(nextMode) {
        if (busy || nextMode === mode) return;

        requestRef.current?.abort();

        // A single-image workflow can only keep one upload.
        // When switching down from Before/After, retain T1 and
        // release any extra browser object URLs.
        if (nextMode === "single" && images.length > 1) {
            setImages((current) => {
                const keep = current.slice(0, 1);
                current.slice(1).forEach((img) => {
                    if (img.url?.startsWith("blob:")) {
                        URL.revokeObjectURL(img.url);
                    }
                });
                return keep;
            });
        }

        setMode(nextMode);
        setMessages([]);
        setFeatures(demoFeatures(scene));
        setError("");
        setSelected("");
        setBounds(null);
        setAutoPairKey("");
        setCatalogPairId("");
    }

    async function run(text = query, catalogOverrides = null) {
        if (!text.trim() || busy) return;

        setError("");
        setBusy(true);
        setQuery("");

        const controller = new AbortController();
        requestRef.current = controller;

        try {
            let result;

            // Satellite catalog workflow: retrieve T1/T2 from STAC + analyze.
            if (mode === "catalog") {
                const loc = catalogOverrides?.catalogLocation ?? catalogLocation;
                const date = catalogOverrides?.beforeDate ?? beforeDate;
                const cloud = catalogOverrides?.cloudThreshold ?? cloudThreshold;

                if (!date) {
                    throw new Error("Select a historical date for satellite retrieval.");
                }

                const body = {
                    location: (loc || "").trim() || `${scene.name}, India`,
                    before_date: date,
                    cloud_threshold: cloud,
                    query: text,
                    min_area: minArea,
                    aoi_radius_km: 2.5,
                };

                const bbox = boundsToStacBbox(bounds);
                if (bbox) body.bbox = bbox;
                if (catalogPairId && !catalogOverrides) body.pair_id = catalogPairId;

                const response = await fetch(
                    `${API_URL}/api/temporal-analysis`,
                    {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify(body),
                        signal: controller.signal,
                    },
                );

                const payload = await response.json().catch(() => ({}));

                if (!response.ok) {
                    throw new Error(
                        payload?.detail ||
                        `Satellite analysis failed (${response.status}).`,
                    );
                }

                result = backendResultToFrontend(payload);

                if (controller.signal.aborted) return;

                if (payload.pair_id) setCatalogPairId(payload.pair_id);

                setMessages((v) => [
                    ...v,
                    {id: crypto.randomUUID(), query: text, result},
                ]);
                setFeatures(empty);
                setTab("assistant");
                return;
            }

            // Real change-detection backend for the manual upload workflow.
            if (mode === "temporal" && images.length === 2) {
                const form = new FormData();

                form.append("before_image", images[0].file, images[0].name);
                form.append("after_image", images[1].file, images[1].name);
                form.append("query", text);
                form.append("min_area", String(minArea));

                const response = await fetch(
                    `${API_URL}/analyze`,
                    {
                        method: "POST",
                        body: form,
                        signal: controller.signal,
                    },
                );

                const payload = await response.json().catch(() => ({}));

                if (!response.ok) {
                    throw new Error(
                        payload?.detail ||
                        `Change analysis failed (${response.status}).`,
                    );
                }

                result = backendResultToFrontend(payload);

                if (controller.signal.aborted) return;

                setMessages((v) => [
                    ...v,
                    {
                        id: crypto.randomUUID(),
                        query: text,
                        result,
                    },
                ]);

                setFeatures(empty);
                setTab("assistant");
                return;
            }

            // Keep the existing visual demo workflows for single-image/SAR
            // until those specialist backends are connected.
            if (images.length && mode !== "catalog") {
                throw new Error(
                    "For manual change analysis, select Manual Before / After and upload two images.",
                );
            }

            await new Promise((resolve, reject) => {
                const id = setTimeout(resolve, 350);

                controller.signal.addEventListener(
                    "abort",
                    () => {
                        clearTimeout(id);
                        reject(
                            new DOMException(
                                "Cancelled",
                                "AbortError",
                            ),
                        );
                    },
                );
            });

            result = getDemoResult(
                text,
                mode,
                scene,
            );

            if (controller.signal.aborted) return;

            setMessages((v) => [
                ...v,
                {
                    id: crypto.randomUUID(),
                    query: text,
                    result,
                },
            ]);

            setFeatures(result.features);
            setTab("assistant");

        } catch (e) {

            if (!controller.signal.aborted) {
                setError(
                    e instanceof Error
                        ? e.message
                        : "Analysis failed.",
                );

                setQuery(text);
            }

        } finally {

            if (requestRef.current === controller) {
                setBusy(false);
            }

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

    async function loadDemoPair() {
        if (busy || uploading) return;
        setUploading(true);
        setError("");
        try {
            const [beforeResponse, afterResponse] = await Promise.all([
                fetch("/demo_data/before_demo.png"),
                fetch("/demo_data/after_demo.png"),
            ]);
            if (!beforeResponse.ok || !afterResponse.ok) {
                throw new Error("Demo pair files could not be loaded.");
            }
            const beforeBlob = await beforeResponse.blob();
            const afterBlob = await afterResponse.blob();
            const beforeFile = new File([beforeBlob], "before_demo.png", { type: "image/png" });
            const afterFile = new File([afterBlob], "after_demo.png", { type: "image/png" });
            const before = await readImage(beforeFile, true);
            const after = await readImage(afterFile, true);
            before.date = "2025-01-15";
            after.date = "2025-07-15";
            before.modality = "optical";
            after.modality = "optical";
            setMode("temporal");
            setImages([before, after]);
            setImageIndex(0);
            setMessages([]);
            setFeatures(empty);
            setSelected("");
            setError("");
            setAutoPairKey("");
            setModal(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : "Could not load the demo pair.");
        } finally {
            setUploading(false);
        }
    }

    async function upload(list) {
        if (!list?.length) return;

        setUploading(true);
        setError("");

        const added = [];
        try {
            const limit = mode === "single" ? 1 : 2;
            if (images.length + list.length > limit) throw new Error(`This accepts ${limit} image${limit === 1 ? "" : "s"}. Remove an image or change workflow.`,);
            for (const file of Array.from(list)) added.push(await readImage(file, true));

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

    const suggestions = mode === "catalog"
        ? [
            "What changed between these dates?",
            "What are the vegetation changes?",
            "What are the changes in buildings?",
            "What changed in the water?",
            "What are the terrain changes?",
        ]
        : mode === "single"
            ? ["Highlight the water bodies", "Describe the land cover", "Where are the built-up regions?"]
            : mode === "temporal"
                ? [
                    "What changed between these images?",
                    "What are the changes in buildings?",
                    "What are the vegetation changes?",
                    "What are the terrain changes?",
                    "What changed in the water?",
                ]
                : ["Use optical and SAR to identify water", "Describe the combined scene"];

    const activePair = (() => {
        if (last?.analysisSource === "satellite_catalog" && last?.visuals?.before_png) {
            return {
                before: {
                    name: last.before?.sensor || "Sentinel-2 L2A",
                    date: last.before?.date,
                    subtitle: last.before?.datetime_utc,
                    cloud: last.before?.cloud_cover,
                    url: backendImageSrc(last.visuals.before_png),
                },
                after: {
                    name: "Latest available acquisition",
                    date: last.after?.date,
                    subtitle: last.after?.datetime_utc,
                    cloud: last.after?.cloud_cover,
                    url: backendImageSrc(last.visuals.after_png),
                },
            };
        }
        if (mode === "temporal" && images.length === 2) {
            return {
                before: {name: images[0].name, date: images[0].date, url: images[0].url},
                after: {name: images[1].name, date: images[1].date, url: images[1].url},
            };
        }
        return null;
    })();

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
                                disabled={busy || (images.length > 0 && mode !== "catalog")}
                                value={scene.id}
                                onChange={(e) => handleSceneChange(scenes.find((s) => s.id === e.target.value))}
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
                    <button
                        className="upload-button demo-sidebar-button"
                        disabled={busy || uploading}
                        onClick={loadDemoPair}
                    >
                        <History size={15} /> Load demo pair
                    </button>
                    {mode === "catalog" && (
                        <button
                            className="upload-button demo-sidebar-button"
                            disabled={busy}
                            onClick={() => {
                                const loc = `${scene.name}, India`;
                                const date = "2024-01-15";
                                setCatalogLocation(loc);
                                setBeforeDate(date);
                                setCloudThreshold(20);
                                setCatalogPairId("");
                                run("What are the vegetation changes?", {
                                    catalogLocation: loc,
                                    beforeDate: date,
                                    cloudThreshold: 20,
                                });
                            }}
                        >
                            <Globe2 size={15} /> Live satellite demo
                        </button>
                    )}
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
                    <div className="opacity-label">
                        <label htmlFor="min-change">Change region filter</label>
                        <span>{minArea}px</span>
                    </div>
                    <input
                        id="min-change"
                        className="opacity-slider"
                        type="range"
                        min="500"
                        max="10000"
                        step="100"
                        value={minArea}
                        onChange={(e) => setMinArea(+e.target.value)}
                        title="Minimum connected region area"
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
                        {activePair ? (
                            <TemporalComparison
                                pair={activePair}
                                result={last}
                                lastQuery={messages.at(-1)?.query}
                                busy={busy}
                            />
                        ) : (
                            <>
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
                                    <small>Map region</small>
                                </div>)}
                                {images[imageIndex] && !images[imageIndex].bounds && (<div className="unreferenced-preview">
                                    <img
                                        src={images[imageIndex].url}
                                        alt={images[imageIndex].name}
                                    />
                                    <span>Image preview · no geographic coordinates</span>
                                </div>)}
                                <div className="map-legend">
                                  <span className="legend-title">MAP OVERLAYS</span>
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
                            </>
                        )}
                    </div>
                    {mode !== "single" && mode !== "catalog" && (<div className="compare-strip">
                        <div>
                            <History size={16} />
                            <span>
                  {mode === "temporal" ? "Image comparison" : "Sensor comparison"}
                </span>
                            <small>
                                {images.length === 2 ? "Side-by-side comparison is ready" : "Upload a paired dataset to compare"}
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
                    {mode === "catalog" && last?.before && (<div className="compare-strip">
                        <div>
                            <Globe2 size={16} />
                            <span>Satellite temporal pair</span>
                            <small>
                                T1 {last.before.date} → T2 {last.after?.date} (latest available)
                            </small>
                        </div>
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
                                    disabled={busy}
                                    className={mode === m.id ? "active" : ""}
                                    onClick={() => switchMode(m.id)}
                                >
                                    <m.icon size={15} />
                                    {m.label}
                                </button>))}
                            </div>
                            {mode === "catalog" && (
                                <div className="catalog-form">
                                    <label>
                                        Location
                                        <input
                                            type="text"
                                            value={catalogLocation}
                                            placeholder="Bengaluru, India or 12.9716,77.5946"
                                            disabled={busy}
                                            onChange={(e) => {
                                                setCatalogLocation(e.target.value);
                                                setCatalogPairId("");
                                            }}
                                        />
                                    </label>
                                    <label>
                                        Historical date
                                        <input
                                            type="date"
                                            value={beforeDate}
                                            disabled={busy}
                                            onChange={(e) => {
                                                setBeforeDate(e.target.value);
                                                setCatalogPairId("");
                                            }}
                                        />
                                    </label>
                                    <label>
                                        Cloud cover threshold
                                        <select
                                            value={cloudThreshold}
                                            disabled={busy}
                                            onChange={(e) => {
                                                setCloudThreshold(Number(e.target.value));
                                                setCatalogPairId("");
                                            }}
                                        >
                                            <option value={10}>&lt;10%</option>
                                            <option value={20}>&lt;20%</option>
                                            <option value={30}>&lt;30%</option>
                                            <option value={50}>&lt;50%</option>
                                        </select>
                                    </label>
                                    <p className="catalog-hint">
                                        Retrieves Sentinel-2 L2A for the same AOI: closest scene near the historical date, then the latest available acquisition. Draw an AOI on the map to override the default radius.
                                    </p>
                                    <button
                                        type="button"
                                        className="catalog-analyze-btn"
                                        disabled={busy || !beforeDate}
                                        onClick={() => run("What changed between these dates?")}
                                    >
                                        {busy ? "Searching satellite archive…" : "Retrieve & analyze"}
                                    </button>
                                </div>
                            )}
                            <div className="context-card">
                                <div className="context-icon">
                                    {mode === "catalog" ? <Globe2 size={18} /> : <ScanLine size={18} />}
                                </div>
                                <div>
                                    <strong>
                                        {mode === "catalog"
                                            ? (catalogLocation || scene.name)
                                            : (images.length ? images[0].name : scene.name)}
                                    </strong>
                                    <small>
                                        {mode === "catalog"
                                            ? (beforeDate ? `Historical: ${beforeDate}` : "Location + date workflow")
                                            : (images.length ? "Uploaded imagery" : "Selected scene")}
                                    </small>
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
                                    {m.result.kind === "change" && (
                                        <small>CHANGE INTELLIGENCE</small>
                                    )}
                                </div>
                                <h3>{m.result.title}</h3>
                                <p className="answer">{m.result.answer}</p>

                                {m.result.kind === "change" && (
                                    <>
                                        {m.result.warnings?.map((warning, i) => (
                                            <div className="result-warning" key={i}>
                                                {warning}
                                            </div>
                                        ))}

                                        <div className="result-metrics">
                                            {(m.result.metrics || []).map((x, i) => (
                                                <div key={i}>
                                                    <strong>{x.value}</strong>
                                                    <span>{x.label}</span>
                                                </div>
                                            ))}
                                        </div>

                                        <div className="analysis-image-grid">
                                            {m.result.visuals?.overlay_png && (
                                                <figure className="analysis-image">
                                                    <img
                                                        src={backendImageSrc(m.result.visuals.overlay_png)}
                                                        alt="Detected change overlay"
                                                    />
                                                    <figcaption>Detected change regions</figcaption>
                                                </figure>
                                            )}

                                            {m.result.visuals?.heatmap_png && (
                                                <figure className="analysis-image">
                                                    <img
                                                        src={backendImageSrc(m.result.visuals.heatmap_png)}
                                                        alt="Change heatmap"
                                                    />
                                                    <figcaption>Temporal difference heatmap</figcaption>
                                                </figure>
                                            )}

                                            {m.result.visuals?.semantic_overlay_png && (
                                                <figure className="analysis-image analysis-image-wide">
                                                    <img
                                                        src={backendImageSrc(m.result.visuals.semantic_overlay_png)}
                                                        alt="Semantic category change overlay"
                                                    />
                                                    <figcaption>Category-specific change map</figcaption>
                                                </figure>
                                            )}
                                        </div>

                                        {m.result.semantic && (
                                            <div className="semantic-summary">
                                                <div>
                                                    <span>Category</span>
                                                    <strong>{m.result.semantic.category}</strong>
                                                </div>
                                                <div>
                                                    <span>New area</span>
                                                    <strong>{m.result.semantic.new_percentage?.toFixed?.(2) || "0.00"}%</strong>
                                                </div>
                                                <div>
                                                    <span>Lost area</span>
                                                    <strong>{m.result.semantic.removed_percentage?.toFixed?.(2) || "0.00"}%</strong>
                                                </div>
                                                <div>
                                                    <span>Semantic signal confidence</span>
                                                    <strong>{((m.result.semantic.confidence || 0) * 100).toFixed(1)}%</strong>
                                                </div>
                                            </div>
                                        )}

                                        {m.result.disclaimer && (
                                            <p className="result-disclaimer">
                                                {m.result.disclaimer}
                                            </p>
                                        )}

                                        {m.result.analysisSource === "satellite_catalog" && (
                                            <DataProvenance result={m.result} />
                                        )}

                                        <TechnicalDetails result={m.result} />
                                    </>
                                )}

                                {m.result.kind !== "change" && (
                                    <>
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
                                            <Layers3 size={14} /> Show spatial evidence <ArrowRight size={14} />
                                        </button>
                                    </>
                                )}

                                {m.result.kind === "change" && (
                                    <button
                                        className="view-evidence"
                                        onClick={() => {
                                            const visual = m.result.visuals?.semantic_overlay_png ||
                                                m.result.visuals?.overlay_png;
                                            if (!visual) return;
                                            const a = document.createElement("a");
                                            a.href = backendImageSrc(visual);
                                            a.download = "satquery-change-analysis.png";
                                            a.click();
                                        }}
                                    >
                                        <Download size={14} /> Download visual evidence <ArrowRight size={14} />
                                    </button>
                                )}
                            </div>))}
                            {busy && (<div className="thinking" role="status">
                                <LoaderCircle size={15} className="spin" />
                                {mode === "catalog" ? "Searching satellite archive and analyzing change…" : "Preparing results…"}
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
                  {mode === "catalog" ? "Location + date" : mode === "single" ? "Single image" : mode === "temporal" ? "Manual Before / After" : "Optical + SAR"}
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
                            {mode === "single" ? "one image" : "two images of the same area"}{" "}
                            for viewing and AI analysis.
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
                        <div className="demo-pair-callout">
                            <div>
                                <strong>Need a known-good demo?</strong>
                                <span>Load the included controlled Before/After pair with a visible building addition and vegetation-clearance change.</span>
                            </div>
                            <button type="button" className="secondary demo-pair-button" disabled={uploading} onClick={loadDemoPair}>
                                {uploading ? "Loading…" : "Load demo pair"}
                            </button>
                        </div>
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
                                <strong>Location + date</strong>Enter a place and historical date.
                                SatQuery retrieves Sentinel-2 for that date and the latest available
                                acquisition of the same AOI, then runs change analysis.
                            </li>
                            <li>
                                <strong>Manual upload</strong>Upload a Before/After pair, or load
                                the controlled demo images.
                            </li>
                            <li>
                                <strong>Single image / SAR</strong>Existing scene-exploration
                                workflows remain available.
                            </li>
                            <li>
                                <strong>Inspect & export</strong>Review provenance, overlays, and
                                download the report or GeoJSON.
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
