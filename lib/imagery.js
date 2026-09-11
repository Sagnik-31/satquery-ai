import { fromArrayBuffer } from "geotiff";
import proj4 from "proj4";

export async function readImage(
    file,
    benchmark,
) {
    if (file.size > 30 * 1024 * 1024)
        throw new Error("Use an image smaller than 30 MB.");
    const isTiff = /\.tiff?$/i.test(file.name);
    if (!isTiff && !/\.(png|jpe?g)$/i.test(file.name))
        throw new Error("Choose GeoTIFF/TIFF, or a benchmark PNG/JPEG.");
    if (!isTiff && !benchmark)
        throw new Error(
            "PNG/JPEG is supported only for approved benchmark images. Enable the benchmark option.",
        );
    let url = "",
        width = 0,
        height = 0,
        bounds,
        crs = "Unreferenced";
    if (isTiff) {
        const tiff = await fromArrayBuffer(await file.arrayBuffer());
        const image = await tiff.getImage();
        width = image.getWidth();
        height = image.getHeight();
        if (width * height > 100000000)
            throw new Error(
                "This image is too large for browser preview. Use a cropped image under 100 megapixels.",
            );
        const keys = image.getGeoKeys();
        const epsg = keys?.ProjectedCSTypeGeoKey || keys?.GeographicTypeGeoKey;
        if (epsg) {
            crs = `EPSG:${epsg}`;
            if (epsg >= 32601 && epsg <= 32660)
                proj4.defs(
                    crs,
                    `+proj=utm +zone=${epsg - 32600} +datum=WGS84 +units=m +no_defs`,
                );
            if (epsg >= 32701 && epsg <= 32760)
                proj4.defs(
                    crs,
                    `+proj=utm +zone=${epsg - 32700} +south +datum=WGS84 +units=m +no_defs`,
                );
            if (!proj4.defs(crs))
                throw new Error(
                    `Preview does not support ${crs}.`,
                );
            try {
                const box = image.getBoundingBox();
                const corners = [
                    [box[0], box[1]],
                    [box[0], box[3]],
                    [box[2], box[1]],
                    [box[2], box[3]],
                ].map((p) => proj4(crs, "EPSG:4326", p));
                bounds = [
                    [
                        Math.min(...corners.map((p) => p[1])),
                        Math.min(...corners.map((p) => p[0])),
                    ],
                    [
                        Math.max(...corners.map((p) => p[1])),
                        Math.max(...corners.map((p) => p[0])),
                    ],
                ];
            } catch {
                throw new Error("Could not read the geospatial extent.");
            }
            if (
                bounds.flat().some((n) => !Number.isFinite(n)) ||
                bounds[0][0] < -90 ||
                bounds[1][0] > 90 ||
                bounds[0][1] < -180 ||
                bounds[1][1] > 180
            )
                throw new Error(
                    "Image extent is outside valid geographic coordinates.",
                );
        }
        const scale = Math.min(1, 1200 / Math.max(width, height));
        const w = Math.max(1, Math.round(width * scale)),
            h = Math.max(1, Math.round(height * scale));
        const channels = Math.min(3, image.getSamplesPerPixel());
        const samples = await image.readRasters({
            width: w,
            height: h,
            samples: Array.from({length: channels}, (_, i) => i),
        });
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        const pixels = ctx.createImageData(w, h);
        const ranges = Array.from({length: channels}, (_, c) => {
            const vals = Array.from(samples[c])
                .filter(Number.isFinite)
                .sort((a, b) => a - b);
            return [
                vals[Math.floor(vals.length * 0.02)] ?? 0,
                vals[Math.floor(vals.length * 0.98)] ?? 255,
            ];
        });
        for (let i = 0; i < w * h; i++) {
            for (let c = 0; c < 3; c++) {
                const ch = Math.min(c, channels - 1);
                const [min, max] = ranges[ch];
                pixels.data[i * 4 + c] = Math.max(
                    0,
                    Math.min(
                        255,
                        ((Number(samples[ch][i]) - min) / (max - min || 1)) * 255,
                    ),
                );
            }
            pixels.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(pixels, 0, 0);
        url = canvas.toDataURL("image/png");
    } else {
        url = URL.createObjectURL(file);
        try {
            const bitmap = await createImageBitmap(file);
            width = bitmap.width;
            height = bitmap.height;
            bitmap.close();
        } catch {
            URL.revokeObjectURL(url);
            throw new Error("This image could not be decoded.");
        }
    }
    return {
        id: crypto.randomUUID(),
        name: file.name,
        file,
        url,
        width,
        height,
        bounds,
        crs,
        modality: "optical",
        date: "",
        benchmark,
    };
}
