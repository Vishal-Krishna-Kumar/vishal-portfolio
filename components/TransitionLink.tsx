'use client';

import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import React, { ComponentProps } from 'react';

interface Props extends ComponentProps<typeof Link> {
  back?: boolean;
  scrollToId?: string;
}

gsap.registerPlugin(useGSAP);

const TransitionLink = ({ href, onClick, children, back = false, scrollToId, ...rest }: Props) => {
  const router = useRouter();
  const { contextSafe } = useGSAP(() => {});

  const handleLinkClick = contextSafe(async (e: React.MouseEvent<HTMLAnchorElement>) => {
    // Allow normal open in new tab / new window
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;

    e.preventDefault();

    gsap.set('.page-transition', { yPercent: 100 });
    gsap.set('.page-transition--inner', { yPercent: 100 });

    const tl = gsap.timeline();

    tl.to('.page-transition', {
      yPercent: 0,
      duration: 0.28,
      ease: 'power2.out',
    });

    tl.eventCallback('onComplete', () => {
      if (back) {
        // ✅ FIX: Back should restore scroll naturally (no forced scrollTo)
        router.back();
        return;
      }

      if (href) {
        router.push(href.toString());
      } else if (onClick) {
        onClick(e);
      }

      // ✅ Only do hash scrolling when you explicitly pass scrollToId
      // and do it instantly (not smooth) to avoid “fast scroll” on Vercel
      if (typeof window !== 'undefined' && scrollToId && href?.toString().includes('#')) {
        setTimeout(() => {
          const el = document.getElementById(scrollToId);
          if (el) el.scrollIntoView({ behavior: 'auto', block: 'start' });
        }, 350);
      }
    });
  });

  return (
    <Link href={href} {...rest} onClick={handleLinkClick}>
      {children}
    </Link>
  );
};

export default TransitionLink;
