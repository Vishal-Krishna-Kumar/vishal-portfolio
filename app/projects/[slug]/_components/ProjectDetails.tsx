'use client';

import { useLayoutEffect, useRef, useState } from 'react';
import parse from 'html-react-parser';
import ArrowAnimation from '@/components/ArrowAnimation';
import TransitionLink from '@/components/TransitionLink';
import { IProject } from '@/types';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/all';
import { ArrowLeft, ExternalLink, Github } from 'lucide-react';
import { usePathname } from 'next/navigation';

interface Props {
  project: IProject;
}

gsap.registerPlugin(useGSAP, ScrollTrigger);


import { IProjectImage } from '@/types';

const preloadImages = (images: IProjectImage[]) => {
  return Promise.all(
    images.map((imgObj) =>
      new Promise<void>((resolve) => {
        const img = new Image();
        img.onload = () => resolve();
        img.onerror = () => resolve();
        img.src = imgObj.src;
      })
    )
  );
};

const ProjectDetails = ({ project }: Props) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const pathname = usePathname();
  const [parallaxReady, setParallaxReady] = useState(false);

  // ✅ Always start at top + stable ScrollTrigger setup
  useLayoutEffect(() => {
    if (typeof window === 'undefined') return;

    // Kill any old triggers from previous navigation
    ScrollTrigger.getAll().forEach((t) => t.kill());
    ScrollTrigger.clearScrollMemory?.();

    // Force top BEFORE paint
    window.scrollTo(0, 0);

    const prevScrollRestoration = window.history.scrollRestoration;
    window.history.scrollRestoration = 'manual';

    let cancelled = false;

    (async () => {
      // ✅ Preload real background image URLs
      await preloadImages(project.images);

      if (cancelled) return;

      // One more hard top + refresh after images are ready
      window.scrollTo(0, 0);
      requestAnimationFrame(() => {
        window.scrollTo(0, 0);
        ScrollTrigger.refresh(true);
        setParallaxReady(true);
      });
    })();

    return () => {
      cancelled = true;
      window.history.scrollRestoration = prevScrollRestoration;
      ScrollTrigger.getAll().forEach((t) => t.kill());
    };
  }, [pathname, project.images]);

  // Fade in content
  useGSAP(
    () => {
      if (!containerRef.current) return;

      gsap.set('.fade-in-later', { autoAlpha: 0, y: 30 });

      gsap
        .timeline({ delay: 0.25 })
        .to('.fade-in-later', { autoAlpha: 1, y: 0, stagger: 0.08 });
    },
    { scope: containerRef }
  );

  // ✅ Cheaper than blur: fade + slight scale while pinned (desktop only)
  useGSAP(
    () => {
      if (typeof window === 'undefined') return;
      if (window.innerWidth < 992) return;

      gsap.to('#info', {
        autoAlpha: 0,
        scale: 0.95,
        transformOrigin: 'top center',
        force3D: true,
        scrollTrigger: {
          trigger: '#info',
          start: 'bottom bottom',
          end: 'bottom top',
          pin: true,
          pinSpacing: false,
          scrub: 0.5,
        },
      });
    },
    { scope: containerRef }
  );

  // ✅ Parallax only after images are truly ready
  useLayoutEffect(() => {
    if (!parallaxReady) return;

    const wraps = gsap.utils.toArray<HTMLElement>('#images > div');

    wraps.forEach((wrap, i) => {
      const img = wrap.querySelector<HTMLElement>('.parallax-img');
      if (!img) return;

      gsap.fromTo(
        img,
        { yPercent: i ? -8 : -4 },
        {
          yPercent: 8,
          ease: 'none',
          force3D: true,
          scrollTrigger: {
            trigger: wrap,
            start: i ? 'top bottom' : 'top 70%',
            end: 'bottom top',
            scrub: true,
          },
        }
      );
    });

    ScrollTrigger.refresh(true);
  }, [parallaxReady]);

  return (
    <section className="pt-5 pb-14">
      <div className="container" ref={containerRef}>
        {/* ✅ FIX: Use back navigation (restores previous scroll, no fast scroll animation) */}
        <TransitionLink
          href="#"
          back
          className="mb-16 inline-flex gap-2 items-center group h-12"
        >
          <ArrowLeft className="group-hover:-translate-x-1 group-hover:text-primary transition-all duration-300" />
          Back
        </TransitionLink>

        <div className="top-0 min-h-[calc(100svh-100px)] flex" id="info">
          <div className="relative w-full">
            <div className="flex items-start gap-6 mx-auto mb-10 max-w-[635px]">
              <h1 className="fade-in-later opacity-0 text-4xl md:text-[60px] leading-none font-anton overflow-hidden">
                <span className="inline-block">{project.title}</span>
              </h1>

              <div className="fade-in-later opacity-0 flex gap-2">
                {project.sourceCode && (
                  <a
                    href={project.sourceCode}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="hover:text-primary"
                  >
                    <Github size={30} />
                  </a>
                )}
                {project.liveUrl && (
                  <a
                    href={project.liveUrl}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="hover:text-primary"
                  >
                    <ExternalLink size={30} />
                  </a>
                )}
              </div>
            </div>

            <div className="max-w-[635px] space-y-7 pb-20 mx-auto">
              <div className="fade-in-later">
                <p className="text-muted-foreground font-anton mb-3">Year</p>
                <div className="text-lg">{project.year}</div>
              </div>

              <div className="fade-in-later">
                <p className="text-muted-foreground font-anton mb-3">
                  Tech &amp; Technique
                </p>
                <div className="text-lg">{project.techStack.join(', ')}</div>
              </div>

              <div className="fade-in-later">
                <p className="text-muted-foreground font-anton mb-3">Description</p>
                <div className="text-lg prose-xl markdown-text">
                  {parse(project.description)}
                </div>
              </div>

              {project.role && (
                <div className="fade-in-later">
                  <p className="text-muted-foreground font-anton mb-3">My Role</p>
                  <div className="text-lg">{parse(project.role)}</div>
                </div>
              )}
            </div>

            <ArrowAnimation />
          </div>
        </div>

        <div
          className="fade-in-later relative flex flex-col gap-2 max-w-[800px] mx-auto"
          id="images"
        >
          {project.images.map((imageObj) => (
            <div
              key={imageObj.src}
              className="group relative w-full aspect-[750/400] overflow-hidden bg-background-light mb-6"
            >
              <div
                className="parallax-img absolute inset-0 will-change-transform"
                style={{
                  backgroundImage: `url(${imageObj.src})`,
                  backgroundSize: 'cover',
                  backgroundPosition: 'center',
                  backgroundRepeat: 'no-repeat',
                  transform: 'translate3d(0,0,0) scale(1.08)',
                }}
              />

              <a
                href={imageObj.src}
                target="_blank"
                rel="noreferrer noopener"
                className="absolute top-4 right-4 bg-background/70 text-foreground size-12 inline-flex justify-center items-center transition-all opacity-0 hover:bg-primary hover:text-primary-foreground group-hover:opacity-100"
              >
                <ExternalLink />
              </a>

              {/* Title and Description Overlay */}
              {(imageObj.title || imageObj.description) && (
                <div className="absolute bottom-0 left-0 w-full bg-black/60 text-white p-4 text-left">
                  {imageObj.title && <div className="font-bold text-lg mb-1">{imageObj.title}</div>}
                  {imageObj.description && <div className="text-sm opacity-90">{imageObj.description}</div>}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default ProjectDetails;
