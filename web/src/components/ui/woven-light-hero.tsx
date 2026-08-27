"use client";

import React, { useRef, useEffect } from "react";
import { motion, useAnimation } from "framer-motion";
import * as THREE from "three";

export interface WovenLightHeroProps {
  headline?: string;
  subline?: string;
  ctaLabel?: string;
  onCta?: () => void;
  brand?: string;
}

/**
 * Animated hero. Text is prop-driven so the same visual serves any product.
 *
 * Three fixes over the reference implementation, all of which matter once this
 * is actually shipped rather than demoed:
 *
 * 1. The animation loop allocated three THREE.Vector3 per particle per frame -
 *    150,000 objects at 60fps. That is a garbage-collection stall every few
 *    frames. Scratch vectors are now hoisted and reused, and the maths is done
 *    on the raw Float32Array. Identical output, no allocation.
 * 2. The renderer, geometry and material were never disposed, leaking a WebGL
 *    context on every unmount. Next.js navigation unmounts constantly.
 * 3. prefers-reduced-motion is respected: the scene renders as a still frame
 *    rather than animating.
 */
export const WovenLightHero = ({
  headline = "ResearchOS",
  subline = "Ask questions about AI research. Paste a citation you saw in a paper and get it explained in plain language, grounded in the paper itself.",
  ctaLabel = "Start reading",
  onCta,
  brand = "ResearchOS",
}: WovenLightHeroProps) => {
  const textControls = useAnimation();
  const buttonControls = useAnimation();

  useEffect(() => {
    const link = document.createElement("link");
    link.href =
      "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;600&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);

    textControls.start((i: number) => ({
      opacity: 1,
      y: 0,
      transition: { delay: i * 0.1 + 1.5, duration: 1.2, ease: [0.2, 0.65, 0.3, 0.9] },
    }));
    buttonControls.start({ opacity: 1, transition: { delay: 2.5, duration: 1 } });

    return () => {
      if (link.parentNode) document.head.removeChild(link);
    };
  }, [textControls, buttonControls]);

  return (
    <div className="relative flex h-screen w-full flex-col items-center justify-center overflow-hidden bg-black">
      <WovenCanvas />
      <HeroNav brand={brand} />
      <div className="relative z-10 px-4 text-center">
        <h1
          className="text-6xl text-white md:text-8xl"
          style={{
            fontFamily: "'Playfair Display', serif",
            textShadow: "0 0 50px rgba(255, 255, 255, 0.3)",
          }}
        >
          {headline.split(" ").map((word, i) => (
            <span key={i} className="inline-block">
              {word.split("").map((char, j) => (
                <motion.span
                  key={j}
                  custom={i * 5 + j}
                  initial={{ opacity: 0, y: 50 }}
                  animate={textControls}
                  style={{ display: "inline-block" }}
                >
                  {char}
                </motion.span>
              ))}
              {i < headline.split(" ").length - 1 && <span>&nbsp;</span>}
            </span>
          ))}
        </h1>

        <motion.p
          custom={headline.length}
          initial={{ opacity: 0, y: 30 }}
          animate={textControls}
          className="mx-auto mt-6 max-w-xl text-lg text-slate-300"
          style={{ fontFamily: "'Inter', sans-serif" }}
        >
          {subline}
        </motion.p>

        <motion.div initial={{ opacity: 0 }} animate={buttonControls} className="mt-10">
          <button
            onClick={onCta}
            className="rounded-full border-2 border-white/20 bg-white/10 px-8 py-3 font-semibold text-white backdrop-blur-sm transition-all hover:bg-white/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/60"
            style={{ fontFamily: "'Inter', sans-serif" }}
          >
            {ctaLabel}
          </button>
        </motion.div>
      </div>
    </div>
  );
};

const HeroNav = ({ brand }: { brand: string }) => (
  <motion.nav
    initial={{ opacity: 0 }}
    animate={{ opacity: 1, transition: { delay: 1, duration: 1 } }}
    className="absolute top-0 right-0 left-0 z-20 p-6"
  >
    <div className="mx-auto flex max-w-7xl items-center justify-between">
      <div className="flex items-center gap-2">
        <span className="text-2xl font-bold text-white">⎎</span>
        <span
          className="text-xl font-bold text-white"
          style={{ fontFamily: "'Inter', sans-serif" }}
        >
          {brand}
        </span>
      </div>
      <a
        href="https://arxiv.org/list/cs.LG/recent"
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm text-white/60 transition-colors hover:text-white"
        style={{ fontFamily: "'Inter', sans-serif" }}
      >
        arXiv ↗
      </a>
    </div>
  </motion.nav>
);

const WovenCanvas = () => {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000,
    );
    camera.position.z = 5;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const mouse = new THREE.Vector2(0, 0);
    const clock = new THREE.Clock();

    const particleCount = 50000;
    const positions = new Float32Array(particleCount * 3);
    const originalPositions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const velocities = new Float32Array(particleCount * 3);

    const geometry = new THREE.BufferGeometry();
    const torusKnot = new THREE.TorusKnotGeometry(1.5, 0.5, 200, 32);
    const knotPositions = torusKnot.attributes.position;
    const color = new THREE.Color();

    for (let i = 0; i < particleCount; i++) {
      const v = i % knotPositions.count;
      const i3 = i * 3;
      positions[i3] = originalPositions[i3] = knotPositions.getX(v);
      positions[i3 + 1] = originalPositions[i3 + 1] = knotPositions.getY(v);
      positions[i3 + 2] = originalPositions[i3 + 2] = knotPositions.getZ(v);

      color.setHSL(0.55 + Math.random() * 0.25, 0.75, 0.55);
      colors[i3] = color.r;
      colors[i3 + 1] = color.g;
      colors[i3 + 2] = color.b;
    }
    torusKnot.dispose();

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.02,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    const handleMouseMove = (event: MouseEvent) => {
      mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    };
    window.addEventListener("mousemove", handleMouseMove);

    let frameId = 0;
    const INFLUENCE = 1.5;

    const animate = () => {
      frameId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      // Scalar maths on the raw arrays - no per-particle object allocation.
      const mx = mouse.x * 3;
      const my = mouse.y * 3;

      for (let i = 0; i < particleCount; i++) {
        const ix = i * 3;
        const iy = ix + 1;
        const iz = ix + 2;

        const px = positions[ix];
        const py = positions[iy];
        const pz = positions[iz];

        let vx = velocities[ix];
        let vy = velocities[iy];
        let vz = velocities[iz];

        const dx = px - mx;
        const dy = py - my;
        const dz = pz;
        const distSq = dx * dx + dy * dy + dz * dz;

        if (distSq < INFLUENCE * INFLUENCE) {
          const dist = Math.sqrt(distSq) || 1e-6;
          const force = ((INFLUENCE - dist) * 0.01) / dist;
          vx += dx * force;
          vy += dy * force;
          vz += dz * force;
        }

        vx = (vx + (originalPositions[ix] - px) * 0.001) * 0.95;
        vy = (vy + (originalPositions[iy] - py) * 0.001) * 0.95;
        vz = (vz + (originalPositions[iz] - pz) * 0.001) * 0.95;

        positions[ix] = px + vx;
        positions[iy] = py + vy;
        positions[iz] = pz + vz;

        velocities[ix] = vx;
        velocities[iy] = vy;
        velocities[iz] = vz;
      }

      geometry.attributes.position.needsUpdate = true;
      points.rotation.y = elapsed * 0.05;
      renderer.render(scene, camera);
    };

    if (reduceMotion) {
      renderer.render(scene, camera);
    } else {
      animate();
    }

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("mousemove", handleMouseMove);
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
  }, []);

  return <div ref={mountRef} className="absolute inset-0 z-0" />;
};
