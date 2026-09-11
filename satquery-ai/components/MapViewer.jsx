"use client";
import { useEffect, useRef, useState } from "react";
import { Crosshair, Map as MapIcon, Minus, Plus, SquareDashedMousePointer } from "lucide-react";

export const colors = {
    water: "#58c8f2",
    urban: "#f6b96a",
    vegetation: "#a7df91",
};
export default function MapViewer(p) {
    const host = useRef(null),
        map = useRef(null),
        lib = useRef(null),
        tile = useRef(null),
        overlay = useRef(null),
        raster = useRef(null),
        aoi = useRef(null);
    const [ready, setReady] = useState(false),
        [tileError, setTileError] = useState(false),
        [coords, setCoords] = useState(p.scene.center);
    useEffect(() => {
        let cancelled = false;
        import("leaflet").then((L) => {
            if (cancelled || !host.current) return;
            lib.current = L;
            const m = L.map(host.current, {
                zoomControl: false,
                attributionControl: true,
                minZoom: 3,
                maxZoom: 19,
            }).setView(p.scene.center, p.scene.zoom);
            map.current = m;
            L.control.scale({position: "bottomleft", imperial: false}).addTo(m);
            m.on("mousemove", (e) => setCoords([e.latlng.lat, e.latlng.lng]));
            const ro = new ResizeObserver(() => m.invalidateSize());
            ro.observe(host.current);
            (m).dispose = () => ro.disconnect();
            setReady(true);
        });
        return () => {
            cancelled = true;
            (map.current)?.dispose?.();
            map.current?.remove();
            map.current = null;
        };
    }, []);
    useEffect(() => {
        if (!ready) return;
        map.current?.flyTo(p.scene.center, p.scene.zoom, {duration: 0.6});
        aoi.current?.remove();
    }, [p.scene, ready, p.focus]);
    useEffect(() => {
        if (!ready || !map.current || !lib.current) return;
        tile.current?.remove();
        setTileError(false);
        const satellite = p.base === "satellite";
        tile.current = lib.current
            .tileLayer(
                satellite
                    ? "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    : "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                {
                    attribution: satellite
                        ? "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
                        : "© <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a>",
                    maxZoom: 19,
                },
            )
            .addTo(map.current);
        tile.current.on("tileerror", () => setTileError(true));
    }, [p.base, ready]);
    useEffect(() => {
        if (!ready || !map.current || !lib.current) return;
        overlay.current?.remove();
        overlay.current = lib.current
            .geoJSON(p.features, {
                filter: (f) =>
                    p.visible[(f.properties?.kind ?? "water")] !== false,
                style: (f) => ({
                    color: colors[f?.properties?.kind] || "#b9f4d2",
                    weight: 2,
                    fillOpacity: p.opacity * 0.36,
                    opacity: p.opacity,
                    dashArray: f?.properties?.kind === "urban" ? "5 4" : undefined,
                }),
                pointToLayer: (_, latlng) =>
                    lib.current.circleMarker(latlng, {radius: 7}),
                onEachFeature: (f, l) => {
                    const name = String(f.properties?.name || "Imported region");
                    l.on("click", () => p.onSelect(name));
                    const el = document.createElement("span");
                    el.textContent = name;
                    l.bindTooltip(el, {sticky: true});
                },
            })
            .addTo(map.current);
    }, [ready, p.features, p.visible, p.opacity, p.onSelect]);
    useEffect(() => {
        if (!ready || !overlay.current || !map.current) return;
        const box = overlay.current.getBounds();
        if (box.isValid())
            map.current.fitBounds(box, {padding: [60, 60], maxZoom: 14});
    }, [ready, p.features]);
    useEffect(() => {
        if (!ready || !map.current || !lib.current) return;
        raster.current?.remove();
        const img = p.images[p.imageIndex];
        if (img?.bounds) {
            raster.current = lib.current
                .imageOverlay(img.url, img.bounds, {opacity: 1})
                .addTo(map.current);
            raster.current.bringToBack();
            map.current.fitBounds(img.bounds);
        }
    }, [ready, p.images, p.imageIndex]);
    useEffect(() => {
        if (!ready || !map.current || !lib.current) return;
        const m = map.current;
        let first = null;
        const click = (e) => {
            if (!p.drawing) return;
            if (!first) {
                first = e.latlng;
                aoi.current?.remove();
            } else {
                const bounds = lib.current.latLngBounds(first, e.latlng);
                aoi.current = lib
                    .current.rectangle(bounds, {
                        color: "#d6f9bc",
                        weight: 2,
                        fillOpacity: 0.05,
                        dashArray: "6 5",
                    })
                    .addTo(m);
                p.onBounds([
                    [bounds.getSouth(), bounds.getWest()],
                    [bounds.getNorth(), bounds.getEast()],
                ]);
                p.setDrawing(false);
                first = null;
            }
        };
        m.on("click", click);
        return () => {
            m.off("click", click);
        };
    }, [ready, p.drawing, p.onBounds, p.setDrawing]);
    return (
        <>
            <div
                ref={host}
                className={`map-canvas ${p.drawing ? "drawing" : ""}`}
                aria-label="Interactive satellite map"
            />
            {!ready && <div className="map-loading">Loading map…</div>}
            {tileError && (
                <div className="tile-error">
                    Some map tiles could not load. Check your connection or switch
                    basemaps.
                </div>
            )}
            <div className="map-tools">
                <button
                    title="Zoom in"
                    aria-label="Zoom in"
                    onClick={() => map.current?.zoomIn()}
                >
                    <Plus size={18} />
                </button>
                <button
                    title="Zoom out"
                    aria-label="Zoom out"
                    onClick={() => map.current?.zoomOut()}
                >
                    <Minus size={18} />
                </button>
                <span />
                <button
                    title="Fit scene"
                    aria-label="Fit scene"
                    onClick={() => map.current?.fitBounds(p.scene.bounds)}
                >
                    <Crosshair size={18} />
                </button>
                <button
                    className={p.drawing ? "chosen" : ""}
                    title="Select area with two corner clicks"
                    aria-label="Select area"
                    onClick={() => p.setDrawing(!p.drawing)}
                >
                    <SquareDashedMousePointer size={18} />
                </button>
            </div>
            {p.drawing && (
                <div className="draw-hint">
                    Click two opposite corners to select an area{" "}
                    <button onClick={() => p.setDrawing(false)}>Cancel</button>
                </div>
            )}
            <div className="coordinates">
                <MapIcon size={12} />
                {Math.abs(coords[0]).toFixed(4)}° {coords[0] < 0 ? "S" : "N"} &nbsp; {Math.abs(coords[1]).toFixed(4)}° {coords[1] < 0 ? "W" : "E"}{" "}
                <span>WGS 84</span>
            </div>
        </>
    );
}
