'use client';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/all';
import React from 'react';

gsap.registerPlugin(ScrollTrigger, useGSAP);

const AboutMe = () => {
    const container = React.useRef<HTMLDivElement>(null);

    useGSAP(
        () => {
            const tl = gsap.timeline({
                scrollTrigger: {
                    id: 'about-me-in',
                    trigger: container.current,
                    start: 'top 70%',
                    end: 'bottom bottom',
                    scrub: 0.5,
                },
            });

            tl.from('.slide-up-and-fade', {
                y: 150,
                opacity: 0,
                stagger: 0.05,
            });
        },
        { scope: container },
    );

    useGSAP(
        () => {
            const tl = gsap.timeline({
                scrollTrigger: {
                    id: 'about-me-out',
                    trigger: container.current,
                    start: 'bottom 50%',
                    end: 'bottom 10%',
                    scrub: 0.5,
                },
            });

            tl.to('.slide-up-and-fade', {
                y: -150,
                opacity: 0,
                stagger: 0.02,
            });
        },
        { scope: container },
    );

    return (
        <section className="pb-section" id="about-me">
            <div className="container" ref={container}>
                <h2 className="text-4xl md:text-6xl font-thin mb-20 slide-up-and-fade">
                    I develop intelligent ML systems with a strong focus on accuracy, 
                    explainability, and robust evaluation, ensuring reliable performance 
                    through rigorous experimentation.
                </h2>

                <p className="pb-3 border-b text-muted-foreground slide-up-and-fade">
                    <strong>This is me.</strong>
                </p>

                <div className="grid md:grid-cols-12 mt-9">
                    <div className="md:col-span-5">
                        <p className="text-5xl slide-up-and-fade">
                            Hi, I&apos;m Vishal 👋
                        </p>
                    </div>
                    <div className="md:col-span-7">
                        <div className="text-lg text-muted-foreground max-w-[450px]">
                            <p className="slide-up-and-fade">
                                I&apos;m a Machine learning developer pursuing master&apos;s in Computer Science
                                specialized in Data Science focused on building intelligent, scalable, and production-ready systems
                                I enjoy translating complex problems into practical, well-designed technical solutions with real-world impact.
                            </p>
                            <p className="mt-3 slide-up-and-fade">
                               Currently, my interests are centered around Generative AI, Large Language Models (LLMs), 
                               Retrieval-Augmented Generation (RAG), and 
                               data-driven system design. 
                               I am currently focused on strengthening my skills and actively looking for Summer 2026 internship opportunities.
                           </p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default AboutMe;
