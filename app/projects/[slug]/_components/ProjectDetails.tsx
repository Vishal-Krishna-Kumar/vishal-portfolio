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

const ProjectDetails = ({ project }: Props) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const pathname = usePathname();
  const [parallaxReady, setParallaxReady] = useState(false);

  // ✅ Always start at top (hard reset, stable even with transitions + refresh)
  useLayoutEffect(() => {
    if (typeof window === 'undefined') return;

    // Always set scroll restoration to manual for this page
    const prevScrollRestoration = window.history.scrollRestoration;
    window.history.scrollRestoration = 'manual';

    // Helper to force scroll to top
    const scrollTopHard = () => {
      window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
    };

    // Wait for all images in #images to load (including cached)
    const images = Array.from(document.querySelectorAll('#images .parallax-img')) as HTMLDivElement[];
    let loaded = 0;
    let ready = false;
    const onReady = () => {
      if (ready) return;
      ready = true;
      ScrollTrigger.getAll().forEach((t) => t.kill());
      ScrollTrigger.clearScrollMemory?.();
      scrollTopHard();
      requestAnimationFrame(() => {
        scrollTopHard();
        requestAnimationFrame(() => {
          scrollTopHard();
          ScrollTrigger.refresh(true);
          setParallaxReady(true);
        });
      });
    };
    if (images.length === 0) {
      onReady();
    } else {
      images.forEach((img) => {
        if ((img as any).complete || img.getAttribute('data-loaded') === 'true') {
          loaded++;
          if (loaded === images.length) onReady();
        } else {
          img.addEventListener('load', () => {
            loaded++;
            if (loaded === images.length) onReady();
          }, { once: true });
          img.addEventListener('error', () => {
            loaded++;
            if (loaded === images.length) onReady();
          }, { once: true });
        }
      });
    }

    window.addEventListener('load', scrollTopHard, { once: true });
    const handleVisibility = () => setTimeout(scrollTopHard, 50);
    document.addEventListener('visibilitychange', handleVisibility);
    setTimeout(scrollTopHard, 200);

    return () => {
      window.removeEventListener('load', scrollTopHard);
      document.removeEventListener('visibilitychange', handleVisibility);
      window.history.scrollRestoration = prevScrollRestoration;
      ScrollTrigger.getAll().forEach((t) => t.kill());
    };
  }, [pathname]);

  // Fade in content
  useGSAP(
    () => {
      if (!containerRef.current) return;

      gsap.set('.fade-in-later', { autoAlpha: 0, y: 30 });

      gsap
        .timeline({ delay: 0.5 })
        .to('.fade-in-later', { autoAlpha: 1, y: 0, stagger: 0.1 });
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

  // Parallax GSAP setup: only run when parallaxReady is true
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
    // Final guarantee: force scroll to top after all GSAP/parallax is initialized
    setTimeout(() => {
      window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
    }, 40);
  }, [parallaxReady, pathname]);

  return (
    <section className="pt-5 pb-14">
      <div className="container" ref={containerRef}>
        <TransitionLink
          href="/#selected-projects"
          scrollToId="selected-projects"
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
          {project.images.map((image) => (
            <div
              key={image}
              className="group relative w-full aspect-[750/400] overflow-hidden bg-background-light"
            >
              <div
                className="parallax-img absolute inset-0 will-change-transform"
                style={{
                  backgroundImage: `url(${image})`,
                  backgroundSize: 'cover',
                  backgroundPosition: 'center',
                  backgroundRepeat: 'no-repeat',
                  transform: 'translate3d(0,0,0) scale(1.08)',
                }}
              />

              <a
                href={image}
                target="_blank"
                rel="noreferrer noopener"
                className="absolute top-4 right-4 bg-background/70 text-foreground size-12 inline-flex justify-center items-center transition-all opacity-0 hover:bg-primary hover:text-primary-foreground group-hover:opacity-100"
              >
                <ExternalLink />
              </a>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default ProjectDetails;
