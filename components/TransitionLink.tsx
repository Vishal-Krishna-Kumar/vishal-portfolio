'use client';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import React, { ComponentProps, useEffect } from 'react';

interface Props extends ComponentProps<typeof Link> {
  back?: boolean;
  scrollToId?: string;
}

gsap.registerPlugin(useGSAP);

const TransitionLink = ({
  href,
  onClick,
  children,
  back = false,
  scrollToId,
  ...rest
}: Props) => {
  const router = useRouter();
  const { contextSafe } = useGSAP(() => {});

  // ✅ Keep default behavior so Back restores previous scroll position
  // (DO NOT set scrollRestoration='manual' globally)
  useEffect(() => {
    // nothing here
  }, []);

  const scrollTopHard = () => {
    window.scrollTo(0, 0);
    requestAnimationFrame(() => window.scrollTo(0, 0));
    requestAnimationFrame(() => window.scrollTo(0, 0));
  };

  const handleLinkClick = contextSafe(async (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();

    gsap.set('.page-transition', { yPercent: 100 });
    gsap.set('.page-transition--inner', { yPercent: 100 });

    const tl = gsap.timeline();

    tl.to('.page-transition', {
      yPercent: 0,
      duration: 0.3,
    });

    tl.eventCallback('onComplete', () => {
      if (back) {
        router.back();
        return;
      }

      if (href) {
        router.push(href.toString());
        // If hash present and scrollToId provided, scroll after navigation
        if (typeof window !== 'undefined' && scrollToId && href.toString().includes('#')) {
          setTimeout(() => {
            const el = document.getElementById(scrollToId);
            if (el) {
              el.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
          }, 400); // Wait for navigation/transition
        }
      } else if (onClick) onClick(e);

      scrollTopHard();
    });
  });

  return (
    <Link href={href} {...rest} onClick={handleLinkClick}>
      {children}
    </Link>
  );
};

export default TransitionLink;
