import { useRouter } from 'next/router';
import Link from 'next/link';

export default function Header() {
  const router = useRouter();
  const activePage = router.pathname;
  return (
    <header>
      <ul id="site-nav">
        <li>
          <Link className={activePage === '/developer' && 'active'} href="/developer">
            Developer
          </Link>
        </li>
        <li>
          <Link className={activePage === '/pricing' && 'active'} href="/pricing">
            Pricing
          </Link>
        </li>
      </ul>
    </header>
  );
}
