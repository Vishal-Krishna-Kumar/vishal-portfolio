'use client';
import SectionTitle from '@/components/SectionTitle';
import { MY_EXPERIENCE } from '@/lib/data';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/all';
import { useRef } from 'react';

gsap.registerPlugin(useGSAP, ScrollTrigger);

const Experiences = () => {
    const containerRef = useRef<HTMLDivElement>(null);

    useGSAP(
        () => {
            const tl = gsap.timeline({
                scrollTrigger: {
                    trigger: containerRef.current,
                    start: 'top 60%',
                    end: 'bottom 50%',
                    toggleActions: 'restart none none reverse',
                    scrub: 1,
                },
            });

            tl.from('.experience-item', {
                y: 50,
                opacity: 0,
                stagger: 0.3,
            });
        },
        { scope: containerRef },
    );

    useGSAP(
        () => {
            const tl = gsap.timeline({
                scrollTrigger: {
                    trigger: containerRef.current,
                    start: 'bottom 50%',
                    end: 'bottom 20%',
                    scrub: 1,
                },
            });

            tl.to(containerRef.current, {
                y: -150,
                opacity: 0,
            });
        },
        { scope: containerRef },
    );

    return (
        <section className="py-section" id="my-experience">
            <div className="container" ref={containerRef}>
                <SectionTitle title="My Experience" />

                <div className="grid gap-14">
                    {MY_EXPERIENCE.map((item) => (
                        <ExperienceItem key={item.title} item={item} />
                    ))}
                </div>
            </div>
        </section>
    );
};


// ExperienceItem component with hover image and description
import { useState } from 'react';
import Image from 'next/image';
const ExperienceItem = ({ item }: { item: any }) => {
    const [hovered, setHovered] = useState(false);
    return (
        <div
            className="experience-item relative"
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
        >
            <p className="text-xl text-muted-foreground">{item.company}</p>
            <div className="inline-block relative">
                <p className="text-5xl font-anton leading-none mt-3.5 mb-2.5 cursor-pointer">
                    {item.title}
                </p>
                {hovered && item.thumbnail && (
                    <Image
                        src={item.thumbnail}
                        alt={item.title}
                        width={224}
                        height={144}
                        className="absolute left-full top-1/2 -translate-y-1/2 ml-6 w-56 h-36 object-cover rounded-lg shadow-lg z-20 border border-neutral-800 bg-neutral-900"
                        style={{ pointerEvents: 'none' }}
                    />
                )}
            </div>
            {/* Description always visible below title */}
            {item.description && (
                <p className="text-base text-neutral-300 mt-3 max-w-2xl">
                    {item.description}
                </p>
            )}
            {/* Duration if present */}
            {item.duration && (
                <p className="text-lg text-muted-foreground mt-2">{item.duration}</p>
            )}
        </div>
    );
};

export default Experiences;
